import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium  # Ensure gymnasium is installed if used

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS acceleration on M4")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU only")

# Define an Actor-Critic network that processes both depth map and IMU data
class ActorCritic(nn.Module):
    def __init__(self, depth_channels, imu_dim, action_dim, dropout_rate=0.2):
        """
        depth_channels: number of channels in the depth map (e.g., 1 for single-channel)
        imu_dim: dimension of the IMU input (e.g., 1 if only vertical acceleration)
        action_dim: number of actions
        """
        super(ActorCritic, self).__init__()
        # Depth branch: convolutional layers to extract features from the depth map
        self.conv_depth = nn.Sequential(
            nn.Conv2d(depth_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduce image size; e.g., from 84x84 to 42x42 (assuming input is 84x84)
        )
        self.flatten_depth = nn.Flatten()
        # Calculate the flattened dimension from depth branch (adjust if input size changes)
        self.depth_feature_dim = 64 * 42 * 42

        # IMU branch: a small fully connected network to process IMU data
        self.fc_imu = nn.Sequential(
            nn.Linear(imu_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.imu_feature_dim = 32

        # Shared fully connected layers after concatenating features
        self.fc_shared = nn.Sequential(
            nn.Linear(self.depth_feature_dim + self.imu_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Actor branch: output action probabilities
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # Softmax over the last dimension to obtain probability distribution
        )

        # Critic branch: output state value
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, depth_input, imu_input):
        """
        depth_input: Tensor with shape (batch_size, depth_channels, H, W)
        imu_input: Tensor with shape (batch_size, imu_dim)
        """
        # Process depth map through convolutional branch
        depth_features = self.conv_depth(depth_input)          # e.g., (batch_size, 64, 42, 42)
        depth_features = self.flatten_depth(depth_features)      # e.g., (batch_size, depth_feature_dim)

        # Process IMU data through fully connected branch
        imu_features = self.fc_imu(imu_input)                    # (batch_size, imu_feature_dim)

        # Concatenate features from both branches along dimension 1
        combined_features = torch.cat([depth_features, imu_features], dim=1)

        # Shared fully connected layers
        shared = self.fc_shared(combined_features)

        # Actor and Critic outputs
        action_probs = self.actor(shared)
        state_value = self.critic(shared)
        return action_probs, state_value


# Memory for storing experiences
class Memory:
    def __init__(self):
        self.states_depth = []    # Stores depth map states
        self.states_imu = []      # Stores IMU states
        self.actions = []         # Stores actions
        self.logprobs = []        # Stores log probabilities of actions
        self.rewards = []         # Stores rewards
        self.is_terminals = []    # Stores terminal flags

    def clear(self):
        self.states_depth = []
        self.states_imu = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


# PPO Agent that uses the above ActorCritic network
class PPO:
    def __init__(self, depth_channels, imu_dim, action_dim, lr=0.002, gamma=0.99, eps_clip=0.2, K_epochs=4):
        # Use the ActorCritic network with two input modalities
        self.policy = ActorCritic(depth_channels, imu_dim, action_dim, dropout_rate=0.2).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(depth_channels, imu_dim, action_dim, dropout_rate=0.2).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, depth_state, imu_state, memory):
        # Set model to eval mode to avoid batchnorm errors on batch_size = 1
        self.policy_old.eval()

        # Preprocess inputs as you already have
        if len(depth_state.shape) == 2:
            depth_state = np.expand_dims(depth_state, axis=0)  # (1, H, W)
            depth_state = np.expand_dims(depth_state, axis=0)  # (1, 1, H, W)
        elif len(depth_state.shape) == 3:
            depth_state = np.transpose(depth_state, (2, 0, 1))
            depth_state = np.expand_dims(depth_state, axis=0)

        if len(imu_state.shape) == 1:
            imu_state = np.expand_dims(imu_state, axis=0)

        depth_state = torch.FloatTensor(depth_state).to(device)
        imu_state = torch.FloatTensor(imu_state).to(device)

        with torch.no_grad():  # Ensure no gradients are tracked
            action_probs, _ = self.policy_old(depth_state, imu_state)
            dist = Categorical(action_probs)
            action = dist.sample()

        memory.states_depth.append(depth_state)
        memory.states_imu.append(imu_state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()


    def update(self, memory):
        self.policy.train()
        old_depth_states = torch.cat(memory.states_depth, dim=0).to(device).detach()  # (batch_size, channels, H, W)
        old_imu_states = torch.cat(memory.states_imu, dim=0).to(device).detach()      # (batch_size, imu_dim)
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Compute discounted cumulative rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        for _ in range(self.K_epochs):
            action_probs, state_values = self.policy(old_depth_states, old_imu_states)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()

            ratios = torch.exp(new_logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach().squeeze()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = self.MseLoss(state_values.squeeze(), rewards)

            loss = loss_actor + 0.5 * loss_critic - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
