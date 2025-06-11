import time
import numpy as np
from wheel_bot_env import WheelBotEnv
from plot import plot_simulation_history, plot_each_separately

# Try to import PPO, ignore if not available
try:
    from PPO import PPO
    from PPO import Memory
    ppo_available = True
except ImportError:
    print("PPO module not available, will use simple control mode")
    ppo_available = False

def run_simple_test(episodes=3, steps_per_episode=400):
    """
    Run environment test using simple predefined action sequences
    """
    env = WheelBotEnv()
    
    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            total_reward = 0
            
            for t in range(steps_per_episode):
                # Test different actions in segments
                if t < 100:
                    action = 0  # Forward
                    print("Action: Forward")
                elif t < 200:
                    action = 1  # Turn left
                    print("Action: Turn left")
                elif t < 300:
                    action = 2  # Turn right
                    print("Action: Turn right")
                else:
                    action = 0  # Forward again
                    print("Action: Forward")
                
                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward
                
                # Slow down simulation speed for observation
                time.sleep(1/60)
                
                if done:
                    break
                
            print(f"Episode {episode} ended, total reward: {total_reward}")
    finally:
        env.close()

def run_ppo_training(episodes=1000, steps_per_episode=200):
    """
    Train robot using PPO algorithm
    """
    if not ppo_available:
        print("PPO module not available, cannot run training")
        return
    
    env = WheelBotEnv()
    
    try:
        # Create PPO agent
        ppo = PPO(depth_channels=1, imu_dim=1, action_dim=3)
        reward_history = []
        linear_acceleration_history = []
        angular_acceleration_history = []
        
        # Training loop
        for episode in range(episodes):
            state_depth, _ = env.reset()
            
            # Get IMU data
            lin_acc, ang_acc = env.get_imu_data()
            # Simplify to a single IMU value (e.g., Z-axis acceleration)
            imu = np.array([lin_acc[2]])
            
            memory = Memory()
            episode_reward = 0
            
            for t in range(steps_per_episode):
                # Choose action based on current state
                action = ppo.select_action(state_depth, imu, memory)
                
                # Execute action
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                
                # Save to memory
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                # Update state
                state_depth = next_state
                lin_acc, ang_acc = env.get_imu_data()
                imu = np.array([lin_acc[2]])
                #print(f"Linear Acceleration: {lin_acc}, Angular Acceleration: {ang_acc}")


                # Store IMU data for analysis
                linear_acceleration_history.append(lin_acc)
                angular_acceleration_history.append(ang_acc)
                
                # Slower simulation speed for observation (can be adjusted or removed during training)
                #time.sleep(1/240)
                
                if done:
                    break
            
            # Update policy at the end of each episode
            ppo.update(memory)
            
            # Save reward history
            reward_history.append(episode_reward)
            print(f"Episode: {episode}, Reward: {episode_reward}, Average Reward: {np.mean(reward_history[-10:])}")
    finally:
        # Plot
        plot_simulation_history(reward_history, linear_acceleration_history, angular_acceleration_history)
        env.close()

def run_custom_control(episodes=3, steps_per_episode=400):
    """
    Implement custom control logic, can be extended as needed
    """
    env = WheelBotEnv()
    
    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            total_reward = 0
            
            for t in range(steps_per_episode):
                # Here you can implement your custom control logic
                # For example, using simple sensor reactive control or other algorithms
                
                # Example: Simple heuristic that selects actions based on depth image
                depth_image = obs
                # Calculate average depth on left and right sides
                left_depth = np.mean(depth_image[:, :42, 0])
                right_depth = np.mean(depth_image[:, 42:, 0])
                
                # Simple obstacle avoidance logic: turn toward the direction with more space (higher value)
                if left_depth > right_depth:
                    action = 1  # Turn left
                elif right_depth > left_depth:
                    action = 2  # Turn right
                else:
                    action = 0  # Forward
                
                print(f"Action: {'Forward' if action == 0 else 'Turn left' if action == 1 else 'Turn right'}")
                
                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward
                obs = next_state
                
                # Slow down simulation speed for observation
                time.sleep(1/60)
                
                if done:
                    break
                
            print(f"Episode {episode} ended, total reward: {total_reward}")
    finally:
        env.close()

if __name__ == "__main__":
    # Select mode to run
    mode = "ppo"
    
    if mode == "simple":
        run_simple_test()
    elif mode == "ppo":
        run_ppo_training(episodes=300, steps_per_episode=150)
    elif mode == "custom":
        run_custom_control()
    else:
        print(f"Unknown mode: {mode}")