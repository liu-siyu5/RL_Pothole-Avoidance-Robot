import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
import math
import random  # 添加随机库

class WheelBotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(5)  # forward, slight left, sharp left, slight right, sharp right
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)  # Ensure using step mode instead of real-time mode

        self.robot = None
        self.terrain_id = None
        self.prev_lin_vel = (0, 0, 0)
        self.prev_ang_vel = (0, 0, 0)
        
        # Wheel joint IDs, matching the URDF
        self.LEFT_WHEEL_JOINT_ID = 0  # base_to_lwheel
        self.RIGHT_WHEEL_JOINT_ID = 1  # base_to_rwheel
        
        # Max speed in radians per second (65 RPM = 65 * 2π/60 rad/s)
        self.MAX_WHEEL_SPEED = 400 * 2 * math.pi / 60  # approximately 6.8 rad/s
        
        # Max torque for N20 motors with 1:150 gear ratio at 6V is ~0.8-1.0 kg·cm
        # Convert to simulation units (Newton-meters)
        # 1 kg·cm = 0.098 N·m, so approximately 0.08-0.1 N·m
        self.MOTOR_MAX_TORQUE = 0.08825985  # N·m

        self.visited = set()
        self.previous_action = None  # 记录上一次的动作
        self.same_action_count = 0  # 记录连续相同动作的次数
        
        # 记录历史动作
        self.action_history = []
        self.max_action_history = 20  # 保存最近20个动作
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # 记录各动作的使用次数
        
        # 强制探索机制
        self.force_exploration = True
        self.exploration_frequency = 10  # 每10步强制一次随机动作
        self.right_turn_bonus = 2.0  # 右转奖励，鼓励使用右转

    def _generate_terrain(self):
        """Generate terrain with random potholes"""
        terrain_size = 300
        terrain_scale = 1
        height_data = np.zeros((terrain_size, terrain_size))

        # Create random potholes
        num_potholes = np.random.randint(500, 1000)
        #num_potholes=0
        for _ in range(num_potholes):
            x, y = np.random.randint(4, terrain_size - 4, size=2)
            # Move the pothole if too close to reset position
            while (x < 4 or x > terrain_size - 4 or y < 4 or y > terrain_size - 4):
                x, y = np.random.randint(4, terrain_size - 4, size=2)
            # Random radius and depth for the pothole
            radius = np.random.randint(1, 3)
            depth = np.random.uniform(-0.001, -0.01)
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if (i**2 + j**2) <= radius**2 * np.random.uniform(0.7, 1.2):
                        if 0 <= x + i < terrain_size and 0 <= y + j < terrain_size:
                            height_data[x + i, y + j] = depth

        # Create terrain
        height_data_flat = height_data.flatten().tolist()
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[0.01, 0.01, terrain_scale],
            heightfieldData=height_data_flat,
            numHeightfieldRows=terrain_size,
            numHeightfieldColumns=terrain_size
        )
        self.terrain_id = p.createMultiBody(0, terrain_shape)
        return self.terrain_id

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # Set PyBullet debug view - appropriate for small robot
        p.resetDebugVisualizerCamera(cameraDistance=0.3, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.03])

        # Generate terrain
        self._generate_terrain()
        
        # Load URDF
        self.robot = p.loadURDF("robot_copy.urdf", [0, 0, 0.01])  # Adjusted start height to match robot scale

        # Set friction properties
        for i in range(p.getNumJoints(self.robot)):
            p.changeDynamics(self.robot, i, lateralFriction=1.0)
        
        # Set base link friction
        p.changeDynamics(self.robot, -1, lateralFriction=1.0)
        
        # Set terrain friction
        p.changeDynamics(self.terrain_id, -1, lateralFriction=1.0)
        
        # Reset velocity record
        self.prev_lin_vel = (0, 0, 0)
        self.prev_ang_vel = (0, 0, 0)

        # Initialize position tracking
        self.start_position = [0, 0, 0]
        self.last_position = [0, 0, 0]
        self.max_distance = 0.01  # Small positive value to avoid division by zero
        self.step_count = 0
        self.previous_heading = [1, 0, 0]  # Start assuming facing +X direction
        
        # 重置动作追踪变量
        self.previous_action = None
        self.same_action_count = 0
        
        # 重置动作历史
        self.action_history = []
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        # Wait for physics to stabilize
        for _ in range(10):
            p.stepSimulation()
            
        return self._get_obs(), {}

    def _get_obs(self):
        """Get observation data (depth image)"""
        # Get the position and orientation of the scanner link
        scanner_link_index = 4
        
        # Get link state returns (position, orientation, ...) of the link
        link_state = p.getLinkState(self.robot, scanner_link_index)
        link_pos = link_state[0]  # Position
        link_orn = link_state[1]  # Orientation (quaternion)
        
        # Use the link's orientation to determine forward direction
        # Convert quaternion to rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(link_orn)
        
        # Extract axes from rotation matrix
        # Each column of the rotation matrix represents an axis in world coordinates
        forward_vec = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]  # First column: x-axis
        up_vec = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]  # Third column: z-axis
        
        # Target is straight ahead along the forward vector
        cam_target = [
            link_pos[0] + forward_vec[0],
            link_pos[1] + forward_vec[1],
            link_pos[2] + forward_vec[2]
        ]
        
        view_matrix = p.computeViewMatrix(link_pos, cam_target, up_vec)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.001, farVal=0.1)
        
        _, _, _, depth_img, _ = p.getCameraImage(84, 84, view_matrix, proj_matrix)
        # Convert depth to float32 instead of uint8 to preserve precision
        # The depth buffer values are between 0 and 1, representing normalized depth
        depth_array = np.array(depth_img, dtype=np.float32).reshape((84, 84, 1))
        
        # Optional: Apply normalization that preserves small differences
        # This rescales the depth values to use the full range while maintaining relative differences
        depth_min = np.min(depth_array)
        depth_max = np.max(depth_array)
        if depth_max > depth_min:
            normalized_depth = (depth_array - depth_min) / (depth_max - depth_min)
            # Convert to observation space range (0-255) with higher precision
            depth_array = (normalized_depth * 255).astype(np.uint8)
        else:
            depth_array = np.zeros((84, 84, 1), dtype=np.uint8)
        
        return depth_array
    
    def step(self, action):
        """Execute one step action, directly specify wheel rotation axis and speed"""
        # 强制探索机制：周期性强制随机动作
        if self.force_exploration and self.step_count % self.exploration_frequency == 0:
            # 20%概率选择右转动作（3或4）
            if random.random() < 0.2:
                action = random.choice([3, 4])  # 强制选择右转
            else:
                action = random.randint(0, 4)  # 完全随机
        
        # 更新连续相同动作计数
        if self.previous_action == action:
            self.same_action_count += 1
        else:
            self.same_action_count = 0
        
        self.previous_action = action
        
        # 记录动作历史
        self.action_history.append(action)
        if len(self.action_history) > self.max_action_history:
            old_action = self.action_history.pop(0)
            self.action_counts[old_action] -= 1
        
        # 更新动作计数
        self.action_counts[action] += 1
        
        # Set wheel speeds based on action
        if action == 0:  # Forward
            left_vel = self.MAX_WHEEL_SPEED
            right_vel = self.MAX_WHEEL_SPEED
        elif action == 1:  # Slight left turn
            left_vel = self.MAX_WHEEL_SPEED * 0.8
            right_vel = self.MAX_WHEEL_SPEED
        elif action == 2:  # Sharp left turn
            left_vel = self.MAX_WHEEL_SPEED * 0.4
            right_vel = self.MAX_WHEEL_SPEED
        elif action == 3:  # Slight right turn
            left_vel = self.MAX_WHEEL_SPEED
            right_vel = self.MAX_WHEEL_SPEED * 0.8
        elif action == 4:  # Sharp right turn
            left_vel = self.MAX_WHEEL_SPEED
            right_vel = self.MAX_WHEEL_SPEED * 0.4
        
        # Use PyBullet's API to set wheel speeds
        p.setJointMotorControl2(
            bodyUniqueId=self.robot,
            jointIndex=self.LEFT_WHEEL_JOINT_ID,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_vel,
            force=self.MOTOR_MAX_TORQUE  # Using appropriate torque value for N20 motors
        )
        
        p.setJointMotorControl2(
            bodyUniqueId=self.robot,
            jointIndex=self.RIGHT_WHEEL_JOINT_ID,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_vel,
            force=self.MOTOR_MAX_TORQUE  # Using appropriate torque value for N20 motors
        )
        
        # Execute multiple simulation steps for more stable physics effects
        for _ in range(5):
            p.stepSimulation()
        
        # Get observation, calculate reward, etc.
        obs = self._get_obs()
        reward = self._compute_reward(action)
        done = False
        
        self.step_count += 1
        
        return obs, reward, done, False, {}


    def _compute_reward(self, action=None):
        """Calculate reward based on forward motion and exploration"""
        # Get linear and angular velocity/acceleration
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        lin_acc, ang_acc = self.get_imu_data(update=False)
        
        # Get current position and orientation
        current_pos, current_orn = p.getBasePositionAndOrientation(self.robot)
        
        # Calculate heading vector (forward direction in world coordinates)
        rotation_matrix = p.getMatrixFromQuaternion(current_orn)
        current_heading = [rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]]
        
        # Calculate distance from start
        distance_from_start = np.sqrt((current_pos[0] - self.start_position[0])**2 + 
                                    (current_pos[1] - self.start_position[1])**2)
        
        # Update max distance if we've gone further
        if distance_from_start > self.max_distance:
            self.max_distance = distance_from_start
            # Add bonus for reaching new max distance
            max_distance_bonus = 0.05
        else:
            max_distance_bonus = -3.0
        
        # Calculate distance moved since last step
        distance_moved = np.sqrt((current_pos[0] - self.last_position[0])**2 + 
                                (current_pos[1] - self.last_position[1])**2)
        
        # Update last position for next step
        self.last_position = current_pos
        
        # REWARDS:
        
        # 1. Forward velocity reward - decreased weight
        forward_velocity_reward = max(0, lin_vel[0])
        
        # 2. Exploration reward - increased reward for new cells and reduced penalty
        grid_size = 0.25  # Smaller grid size for finer exploration
        grid_xy = (int(current_pos[0] // grid_size), int(current_pos[1] // grid_size))
        
        if grid_xy not in self.visited:
            self.visited.add(grid_xy)
            novelty_reward = 10.0  # Increased reward for visiting new cells
        else:
            novelty_reward = -0.2  # Reduced penalty for revisiting
        
        # 3. Direction consistency - reduced impact to allow more turning
        heading_consistency = current_heading[0] * self.previous_heading[0] + current_heading[1] * self.previous_heading[1]
        direction_consistency_reward = heading_consistency * 0.5  # Reduced from 2.0 to 0.5
        self.previous_heading = current_heading
        
        # 4. Distance-based exploration bonus
        distance_bonus = self.max_distance * 0.5  # Reward based on max distance reached
        
        # 5. Reduced time penalty
        time_penalty = -0.1  # Much smaller time penalty
        
        # 6. Smoothness penalty (reduced weight)
        x_mean, x_std = 0.0061, 0.8255
        y_mean, y_std = -0.0032, 1.4097
        z_mean, z_std = 0.0010, 0.6566
        
        x_deviation = abs(ang_acc[0] - x_mean) / x_std
        y_deviation = abs(ang_acc[1] - y_mean) / y_std
        z_deviation = abs(ang_acc[2] - z_mean) / z_std
        
        smoothness_penalty = (
            -0.05 * (1 / (1 + np.exp(-x_deviation + 5))) +
            -0.05 * (1 / (1 + np.exp(-y_deviation + 5))) +
            -0.1 * (1 / (1 + np.exp(-z_deviation + 5)))
        )
        
        # 7. Angular velocity penalty (reduced)
        angular_velocity_penalty = -0.05 * abs(ang_vel[2])
        
        # 8. 添加重复动作惩罚（两种惩罚机制）
        # 8.1 连续相同动作惩罚
        repeated_action_penalty = -0.5 * min(self.same_action_count, 5)  # 限制惩罚上限为-2.5
        
        # 8.2 动作分布不均衡惩罚
        if len(self.action_history) > 0:
            # 计算最常用动作与平均的差距
            avg_count = len(self.action_history) / len(self.action_counts)
            max_count = max(self.action_counts.values())
            action_imbalance_penalty = -0.2 * max(0, (max_count - avg_count) / avg_count)
        else:
            action_imbalance_penalty = 0
            
        # 9. 右转奖励 (鼓励选择右转动作)
        right_turn_bonus = 0
        if action in [3, 4]:  # 右转动作
            right_turn_bonus = self.right_turn_bonus
        
        # Combined reward with adjusted weighting
        reward = (
            forward_velocity_reward +
            novelty_reward +
            direction_consistency_reward +
            smoothness_penalty +
            angular_velocity_penalty +
            time_penalty +
            distance_bonus +
            max_distance_bonus +
            repeated_action_penalty +
            action_imbalance_penalty +
            right_turn_bonus
        )
        
        print(f"Vel: {forward_velocity_reward:.2f}, " +
            f"Explore: {novelty_reward:.2f}, Consist: {direction_consistency_reward:.2f}, " +
            f"Smooth: {smoothness_penalty:.2f}, DistBonus: {distance_bonus:.2f}, " +
            f"MaxDist: {max_distance_bonus:.2f}, " +
            f"RepeatAction: {repeated_action_penalty:.2f}, " +
            f"ActionImbalance: {action_imbalance_penalty:.2f}, " +
            f"RightBonus: {right_turn_bonus:.2f}")
        
        return reward

    def close(self):
        """Close the environment"""
        p.disconnect()

    def get_imu_data(self, update=True):
        """Get IMU data (simulated acceleration and angular velocity)"""
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        
        # Calculate acceleration (through velocity change)
        lin_acc = np.array(lin_vel) - np.array(self.prev_lin_vel)
        ang_acc = np.array(ang_vel) - np.array(self.prev_ang_vel)
        
        # If update is True, update previous frame velocity
        if update:
            self.prev_lin_vel = lin_vel
            self.prev_ang_vel = ang_vel
        
        #print(f"Linear acceleration: {lin_acc}, Angular acceleration: {ang_acc}")
        return lin_acc, ang_acc