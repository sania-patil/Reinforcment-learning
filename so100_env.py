# # # import gym
# # # import pybullet as p
# # # import pybullet_data
# # # import numpy as np
# # # import time
# # # from gym import spaces
# # # import math
# # # import os

# # # class SO100PickEnv(gym.Env):
# # #     def __init__(self, render_mode=False):
# # #         super(SO100PickEnv, self).__init__()
# # #         self.render_mode = render_mode
# # #         self.time_step = 1.0 / 240.0
# # #         self.max_steps = 200
# # #         self.step_counter = 0
# # #         self.urdf_path = "D:/robotics/so100.urdf"

# # #         self._connect()

# # #         # Action: delta x, y, z for gripper in range [-0.05, 0.05]
# # #         self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)

# # #         # Observation: gripper_pos (3), cube_pos (3)
# # #         obs_low = np.array([-1, -1, 0, -1, -1, 0])
# # #         obs_high = np.array([1, 1, 1, 1, 1, 1])
# # #         self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

# # #     def _connect(self):
# # #         if self.render_mode:
# # #             p.connect(p.GUI)
# # #         else:
# # #             p.connect(p.DIRECT)
# # #         p.setAdditionalSearchPath(pybullet_data.getDataPath())
# # #         p.setGravity(0, 0, -9.81)

# # #     def reset(self):
# # #         p.resetSimulation()
# # #         p.setGravity(0, 0, -9.81)

# # #         self.plane = p.loadURDF("plane.urdf")
# # #         self.robot = p.loadURDF(self.urdf_path, useFixedBase=True)

# # #         self.joint_indices = [0, 1, 2, 3]
# # #         self.gripper_left = 4
# # #         self.gripper_right = 5
# # #         self.end_effector_index = 3

# # #         # Random cube position in front of robot
# # #         x = np.random.uniform(0.20, 0.30)
# # #         y = np.random.uniform(-0.05, 0.05)
# # #         self.cube_pos = [x, y, 0.02]
# # #         self.cube_id = p.loadURDF("cube_small.urdf", basePosition=self.cube_pos)

# # #         # Open gripper and move to initial pose
# # #         self._control_gripper(open=True)
# # #         self._move_arm_to([x, y, 0.2])

# # #         self.step_counter = 0
# # #         return self._get_obs()

# # #     def _get_obs(self):
# # #         gripper_pos = p.getLinkState(self.robot, self.end_effector_index)[0]
# # #         cube_pos = p.getBasePositionAndOrientation(self.cube_id)[0]
# # #         return np.array(gripper_pos + cube_pos, dtype=np.float32)

# # #     # def step(self, action):
# # #     #     self.step_counter += 1

# # #     #     # Clip delta movement and get current gripper position
# # #     #     action = np.clip(action, -0.05, 0.05)
# # #     #     current_pos = np.array(p.getLinkState(self.robot, self.end_effector_index)[0])
# # #     #     new_pos = current_pos + action
# # #     #     new_pos = np.clip(new_pos, [0.1, -0.2, 0.02], [0.4, 0.2, 0.3])  # workspace limits

# # #     #     # Move arm
# # #     #     self._move_arm_to(new_pos)

# # #     #     # Check distance to cube
# # #     #     cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
# # #     #     distance = np.linalg.norm(new_pos - cube_pos)

# # #     #     reward = -distance
# # #     #     done = False

# # #     #     # Try to grasp when close enough
# # #     #     if distance < 0.04:
# # #     #         self._control_gripper(open=False)
# # #     #         time.sleep(0.2)
# # #     #         cube_new_pos = p.getBasePositionAndOrientation(self.cube_id)[0]
# # #     #         if cube_new_pos[2] > 0.05:
# # #     #             reward += 10.0  # successful lift
# # #     #             done = True

# # #     #     if self.step_counter >= self.max_steps:
# # #     #         done = True

# # #     #     return self._get_obs(), reward, done, {}
# # # def step(self, action):
# # #     self.step_counter += 1

# # #     # Clip delta movement and get current gripper position
# # #     action = np.clip(action, -0.05, 0.05)
# # #     current_pos = np.array(p.getLinkState(self.robot, self.end_effector_index)[0])
# # #     new_pos = current_pos + action
# # #     new_pos = np.clip(new_pos, [0.1, -0.2, 0.02], [0.4, 0.2, 0.3])  # workspace limits

# # #     # Move arm
# # #     self._move_arm_to(new_pos)

# # #     # Check distance to cube
# # #     cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
# # #     xy_distance = np.linalg.norm(new_pos[:2] - cube_pos[:2])

# # #     reward = -xy_distance
# # #     done = False

# # #     # Try to grasp when XY distance is close enough
# # #     if xy_distance < 0.03:
# # #         # Lower gripper down to grasp height
# # #         grasp_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.01]  # lower than before
# # #         self._move_arm_to(grasp_pos)

# # #         self._control_gripper(open=False)
# # #         time.sleep(0.2)

# # #         # Lift after grasp
# # #         lift_pos = [cube_pos[0], cube_pos[1], 0.2]
# # #         self._move_arm_to(lift_pos)

# # #         # Check if cube was lifted
# # #         cube_new_z = p.getBasePositionAndOrientation(self.cube_id)[0][2]
# # #         if cube_new_z > 0.05:
# # #             reward += 10.0  # successful lift
# # #             done = True

# # #     if self.step_counter >= self.max_steps:
# # #         done = True

# # #     return self._get_obs(), reward, done, {}

# # #     def _move_arm_to(self, position):
# # #         orn = p.getQuaternionFromEuler([0, math.pi, 0])
# # #         joint_angles = p.calculateInverseKinematics(self.robot, self.end_effector_index, position, orn)
# # #         for i, j in enumerate(self.joint_indices):
# # #             p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, joint_angles[i])
# # #         for _ in range(30):
# # #             p.stepSimulation()
# # #             time.sleep(self.time_step if self.render_mode else 0)

# # #     def _control_gripper(self, open=True):
# # #         val = 0.02 if open else 0.0
# # #         force = 50 if not open else 10
# # #         p.setJointMotorControl2(self.robot, self.gripper_left, p.POSITION_CONTROL, targetPosition=val, force=force)
# # #         p.setJointMotorControl2(self.robot, self.gripper_right, p.POSITION_CONTROL, targetPosition=-val, force=force)
# # #         for _ in range(20):
# # #             p.stepSimulation()
# # #             time.sleep(self.time_step if self.render_mode else 0)

# # #     def render(self, mode='human'):
# # #         pass  # already rendered if GUI mode used

# # #     def close(self):
# # #         p.disconnect()


# import gym
# import pybullet as p
# import pybullet_data
# import numpy as np
# import time
# from gym import spaces
# import math
# import os

# class SO100PickEnv(gym.Env):
#     def __init__(self, render_mode=False):
#         super(SO100PickEnv, self).__init__()
#         self.render_mode = render_mode
#         self.time_step = 1.0 / 240.0
#         self.max_steps = 200
#         self.step_counter = 0
#         self.urdf_path = "D:/robotics/so100.urdf"

#         self._connect()

#         # Action: delta x, y, z for gripper
#         self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)

#         # Observation: gripper_pos (3), cube_pos (3)
#         obs_low = np.array([-1, -1, 0, -1, -1, 0])
#         obs_high = np.array([1, 1, 1, 1, 1, 1])
#         self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

#     def _connect(self):
#         if self.render_mode:
#             p.connect(p.GUI)
#         else:
#             p.connect(p.DIRECT)
#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         p.setGravity(0, 0, -9.81)

#     def reset(self):
#         p.resetSimulation()
#         p.setGravity(0, 0, -9.81)

#         self.plane = p.loadURDF("plane.urdf")
#         self.robot = p.loadURDF(self.urdf_path, useFixedBase=True)

#         self.joint_indices = [0, 1, 2, 3]
#         self.gripper_left = 4
#         self.gripper_right = 5
#         self.end_effector_index = 3

#         # Random cube position
#         x = np.random.uniform(0.20, 0.30)
#         y = np.random.uniform(-0.05, 0.05)
#         self.cube_pos = [x, y, 0.02]
#         self.cube_id = p.loadURDF("cube_small.urdf", basePosition=self.cube_pos)
#         p.changeDynamics(self.cube_id, -1, lateralFriction=1.0, rollingFriction=0.01)

#         # Open gripper and move to hover
#         self._control_gripper(open=True)
#         self._move_arm_to([x, y, 0.1])

#         self.step_counter = 0
#         self.has_grasped = False
#         return self._get_obs()

#     def _get_obs(self):
#         gripper_pos = np.array(p.getLinkState(self.robot, self.end_effector_index)[0])
#         cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
        
#         obs = np.concatenate([gripper_pos, cube_pos])
#         return obs.astype(np.float32)


#     # def step(self, action):
#     #     self.step_counter += 1

#     #     action = np.clip(action, -0.05, 0.05)
#     #     current_pos = np.array(p.getLinkState(self.robot, self.end_effector_index)[0])
#     #     new_pos = current_pos + action
#     #     new_pos = np.clip(new_pos, [0.1, -0.2, 0.02], [0.4, 0.2, 0.3])
#     #     self._move_arm_to(new_pos)

#     #     cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
#     #     xy_dist = np.linalg.norm(new_pos[:2] - cube_pos[:2])

#     #     reward = -xy_dist
#     #     done = False

#     #     if not self.has_grasped and xy_dist < 0.03:
#     #         # Try to grasp
#     #         self._move_arm_to([cube_pos[0], cube_pos[1], cube_pos[2] + 0.01])
#     #         self._control_gripper(open=False)
#     #         time.sleep(0.2)
#     #         self._move_arm_to([cube_pos[0], cube_pos[1], 0.2])
#     #         time.sleep(0.2)

#     #         # Check contact
#     #         contacts = p.getContactPoints(bodyA=self.robot, bodyB=self.cube_id)
#     #         if len(contacts) > 0:
#     #             reward += 10.0
#     #             self.has_grasped = True
#     #             done = True

#     #     if self.step_counter >= self.max_steps:
#     #         done = True

#     #     return self._get_obs(), reward, done, {}

#     def step(self, action):
#         self.step_counter += 1

#         # Clip the action to limit movement per step
#         action = np.clip(action, -0.05, 0.05)

#         # Get current end-effector position
#         current_pos = np.array(p.getLinkState(self.robot, self.end_effector_index)[0])
#         new_pos = current_pos + action
#         new_pos = np.clip(new_pos, [0.1, -0.2, 0.02], [0.4, 0.2, 0.3])
#         self._move_arm_to(new_pos)

#         # Get cube position
#         cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
#         xy_dist = np.linalg.norm(new_pos[:2] - cube_pos[:2])
#         z_dist = abs(new_pos[2] - cube_pos[2])

#         reward = -xy_dist  # penalize distance in XY
#         done = False

#         if not self.has_grasped and xy_dist < 0.03 and z_dist < 0.05:
#             # Try to grasp
#             self._move_arm_to([cube_pos[0], cube_pos[1], cube_pos[2] + 0.005])
#             self._control_gripper(open=False)
#             time.sleep(0.2)

#             # Attempt to lift
#             self._move_arm_to([cube_pos[0], cube_pos[1], 0.2])
#             time.sleep(0.2)

#             # Check if cube was lifted
#             lifted_z = p.getBasePositionAndOrientation(self.cube_id)[0][2]
#             if lifted_z > 0.05:  # cube lifted from table
#                 reward += 10.0
#                 self.has_grasped = True
#                 done = True

#         # Episode end condition
#         if self.step_counter >= self.max_steps:
#             done = True

#         return self._get_obs(), reward, done, {}

#     def _move_arm_to(self, position):
#         orn = p.getQuaternionFromEuler([0, math.pi, 0])
#         joint_angles = p.calculateInverseKinematics(self.robot, self.end_effector_index, position, orn)
#         for i, j in enumerate(self.joint_indices):
#             p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, joint_angles[i])
#         for _ in range(30):
#             p.stepSimulation()
#             if self.render_mode:
#                 time.sleep(self.time_step)

#     # def _control_gripper(self, open=True):
#     #     val = 0.04 if open else -0.01
#     #     force = 10 if open else 200
#     #     p.setJointMotorControl2(self.robot, self.gripper_left, p.POSITION_CONTROL, targetPosition=val, force=force)
#     #     p.setJointMotorControl2(self.robot, self.gripper_right, p.POSITION_CONTROL, targetPosition=-val, force=force)
#     #     for _ in range(20):
#     #         p.stepSimulation()
#     #         if self.render_mode:
#     #             time.sleep(self.time_step)

#     def _control_gripper(self, open=True):
#         target = 0.04 if open else 0.0  # Adjust if your gripper closes tighter
#         p.setJointMotorControl2(self.robot, self.gripper_left, p.POSITION_CONTROL, targetPosition=target, force=100)
#         p.setJointMotorControl2(self.robot, self.gripper_right, p.POSITION_CONTROL, targetPosition=target, force=100)
#         for _ in range(10):  # allow some time for fingers to move
#             p.stepSimulation()
#             if self.render_mode:
#                 time.sleep(self.time_step)

#     def render(self, mode='human'):
#         pass

#     def close(self):
#         p.disconnect()



#  

# import gym
# from gym import spaces
# import pybullet as p
# import pybullet_data
# import numpy as np
# import time
# import os

# class CustomGripperEnv(gym.Env):
#     metadata = {"render.modes": ["human", "rgb_array"]}

#     def __init__(self, render=False):
#         super(CustomGripperEnv, self).__init__()

#         self.render_mode = render
#         self.time_step = 1. / 240.
#         self.max_steps = 100
#         self.step_counter = 0

#         if self.render_mode:
#             self.physics_client = p.connect(p.GUI)
#         else:
#             self.physics_client = p.connect(p.DIRECT)

#         p.setTimeStep(self.time_step)
#         p.setGravity(0, 0, -9.8)
#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         self.plane_id = p.loadURDF("plane.urdf")

#         self.robot_start_pos = [0, 0, 0.1]
#         self.robot_id = None
#         self.object_id = None
#         self.reset()

#         # Action: 3D movement (x,y,z), 1 gripper control
#         action_high = np.array([0.05, 0.05, 0.05, 1.0])
#         self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)

#         # Observation: end effector + object position
#         obs_high = np.array([1.0] * 6)
#         self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

#     def reset(self):
#         p.resetSimulation()
#         p.setGravity(0, 0, -9.8)
#         p.setTimeStep(self.time_step)
#         p.setAdditionalSearchPath(pybullet_data.getDataPath())

#         self.plane_id = p.loadURDF("plane.urdf")
#         self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", self.robot_start_pos, useFixedBase=True)

#         # Load simple cube as object
#         object_start_pos = [0.5, 0, 0.1]
#         object_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
#         col_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
#         vis_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025], rgbaColor=[1, 0, 0, 1])
#         self.object_id = p.createMultiBody(baseMass=0.1,
#                                            baseCollisionShapeIndex=col_shape_id,
#                                            baseVisualShapeIndex=vis_shape_id,
#                                            basePosition=object_start_pos,
#                                            baseOrientation=object_start_orientation)

#         self.step_counter = 0
#         return self._get_obs()

#     def step(self, action):
#         dx, dy, dz, gripper = action

#         # Get current EE position
#         ee_link_index = 6  # End effector link
#         state = p.getLinkState(self.robot_id, ee_link_index)
#         current_pos = state[0]

#         # Apply delta movement
#         new_pos = np.clip(np.array(current_pos) + np.array([dx, dy, dz]), [-1, -1, 0], [1, 1, 1])
#         joint_poses = p.calculateInverseKinematics(self.robot_id, ee_link_index, new_pos)

#         for i in range(7):  # Assuming 7 DoF KUKA arm
#             p.setJointMotorControl2(bodyUniqueId=self.robot_id,
#                                     jointIndex=i,
#                                     controlMode=p.POSITION_CONTROL,
#                                     targetPosition=joint_poses[i],
#                                     force=200)

#         # Step simulation
#         for _ in range(5):  # To simulate smooth motion
#             p.stepSimulation()
#             if self.render_mode:
#                 time.sleep(self.time_step)

#         self.step_counter += 1

#         obs = self._get_obs()
#         reward = self._compute_reward(obs)
#         done = self.step_counter >= self.max_steps

#         return obs, reward, done, {}

#     def _get_obs(self):
#         ee_link_index = 6
#         ee_state = p.getLinkState(self.robot_id, ee_link_index)
#         ee_pos = ee_state[0]
#         obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
#         return np.array(ee_pos + obj_pos, dtype=np.float32)

#     def _compute_reward(self, obs):
#         ee_pos = np.array(obs[:3])
#         obj_pos = np.array(obs[3:])
#         dist = np.linalg.norm(ee_pos - obj_pos)
#         reward = -dist
#         if dist < 0.05:
#             reward += 1.0  # bonus for reaching
#         return reward

#     def render(self, mode="human"):
#         pass  # GUI is handled via `render=True`

#     def close(self):
#         p.disconnect()




# import gym
# import pybullet as p
# import pybullet_data
# import numpy as np
# import time
# from gym import spaces
# import math
# import os

# class SO100PickEnv(gym.Env):
#     def __init__(self, render_mode=False):
#         super(SO100PickEnv, self).__init__()
#         self.render_mode = render_mode
#         self.time_step = 1.0 / 240.0
#         self.max_steps = 200
#         self.step_counter = 0
#         self.urdf_path = "D:/robotics/so100.urdf"

#         self._connect()

#         # Action: delta x, y, z for gripper
#         self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)

#         # Observation: gripper_pos (3), cube_pos (3), goal_pos (3)
#         obs_low = np.array([-1, -1, 0, -1, -1, 0, -1, -1, 0])
#         obs_high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
#         self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

#         self.goal_pos = [0.35, 0.1, 0.02]
#         self.has_grasped = False
#         self.has_placed = False

#     def _connect(self):
#         if self.render_mode:
#             p.connect(p.GUI)
#         else:
#             p.connect(p.DIRECT)
#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         p.setGravity(0, 0, -9.81)

#     def reset(self):
#         p.resetSimulation()
#         p.setGravity(0, 0, -9.81)

#         self.plane = p.loadURDF("plane.urdf")
#         self.robot = p.loadURDF(self.urdf_path, useFixedBase=True)

#         self.joint_indices = [0, 1, 2, 3]
#         self.gripper_left = 4
#         self.gripper_right = 5
#         self.end_effector_index = 3

#         # Random cube position
#         x = np.random.uniform(0.20, 0.30)
#         y = np.random.uniform(-0.05, 0.05)
#         self.cube_pos = [x, y, 0.02]
#         self.cube_id = p.loadURDF("cube_small.urdf", basePosition=self.cube_pos)

#         # Fixed goal marker
#         self.goal_id = p.loadURDF("cube_small.urdf", basePosition=self.goal_pos, useFixedBase=True)
#         p.changeVisualShape(self.goal_id, -1, rgbaColor=[1, 0, 0, 0.5])  # red transparent

#         # Initial state
#         self._control_gripper(open=True)
#         self._move_arm_to([x, y, 0.04])

#         self.has_grasped = False
#         self.has_placed = False
#         self.step_counter = 0
#         return self._get_obs()

#     def _get_obs(self):
#         gripper_pos = p.getLinkState(self.robot, self.end_effector_index)[0]
#         cube_pos = p.getBasePositionAndOrientation(self.cube_id)[0]
#         return np.array(gripper_pos + cube_pos + self.goal_pos, dtype=np.float32)

#     def step(self, action):
#         self.step_counter += 1
#         action = np.clip(action, -0.05, 0.05)

#         current_pos = np.array(p.getLinkState(self.robot, self.end_effector_index)[0])
#         new_pos = current_pos + action
#         new_pos = np.clip(new_pos, [0.1, -0.2, 0.02], [0.4, 0.2, 0.3])
#         self._move_arm_to(new_pos)

#         cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
#         goal_pos = np.array(self.goal_pos)

#         reward = -np.linalg.norm(current_pos[:2] - cube_pos[:2])
#         done = False

#         if not self.has_grasped:
#             xy_distance = np.linalg.norm(new_pos[:2] - cube_pos[:2])
#             if xy_distance < 0.03:
#                 grasp_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.005]
#                 self._move_arm_to(grasp_pos)
#                 self._control_gripper(open=False)
#                 time.sleep(0.2)
#                 self._move_arm_to([cube_pos[0], cube_pos[1], 0.2])
#                 lifted_z = p.getBasePositionAndOrientation(self.cube_id)[0][2]
#                 if lifted_z > 0.05:
#                     reward += 5.0
#                     self.has_grasped = True
#         elif not self.has_placed:
#             place_xy_dist = np.linalg.norm(current_pos[:2] - goal_pos[:2])
#             reward = -place_xy_dist
#             if place_xy_dist < 0.03:
#                 place_pos = [goal_pos[0], goal_pos[1], 0.02]
#                 self._move_arm_to(place_pos)
#                 self._control_gripper(open=True)
#                 time.sleep(0.2)
#                 self._move_arm_to([goal_pos[0], goal_pos[1], 0.2])
#                 cube_final_pos = p.getBasePositionAndOrientation(self.cube_id)[0]
#                 if np.linalg.norm(np.array(cube_final_pos[:2]) - goal_pos[:2]) < 0.03:
#                     reward += 10.0
#                     self.has_placed = True
#                     done = True

#         if self.step_counter >= self.max_steps:
#             done = True

#         return self._get_obs(), reward, done, {}

#     def _move_arm_to(self, position):
#         orn = p.getQuaternionFromEuler([0, math.pi, 0])
#         joint_angles = p.calculateInverseKinematics(self.robot, self.end_effector_index, position, orn)
#         for i, j in enumerate(self.joint_indices):
#             p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, joint_angles[i])
#         for _ in range(30):
#             p.stepSimulation()
#             time.sleep(self.time_step if self.render_mode else 0)

#     def _control_gripper(self, open=True):
#         val = 0.02 if open else 0.0
#         force = 50 if not open else 10
#         p.setJointMotorControl2(self.robot, self.gripper_left, p.POSITION_CONTROL, targetPosition=val, force=force)
#         p.setJointMotorControl2(self.robot, self.gripper_right, p.POSITION_CONTROL, targetPosition=-val, force=force)
#         for _ in range(20):
#             p.stepSimulation()
#             time.sleep(self.time_step if self.render_mode else 0)

#     def render(self, mode='human'):
#         pass

#     def close(self):
#         p.disconnect()
# File: envs/so100_env.py

# import gym
# from gym import spaces
# import pybullet as p
# import pybullet_data
# import numpy as np
# import time
# import os

# class SO100PickEnv(gym.Env):
#     def __init__(self, render=False):
#         super(SO100PickEnv, self).__init__()
#         self.render = render
#         if self.render:
#             self.physics_client = p.connect(p.GUI)
#         else:
#             self.physics_client = p.connect(p.DIRECT)

#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         p.setGravity(0, 0, -9.8)
#         self.time_step = 1. / 240.
#         p.setTimeStep(self.time_step)

#         self.max_steps = 200
#         self.step_counter = 0

#         self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)
#         # Observation: gripper xyz + cube xyz
#         self.observation_space = spaces.Box(low=np.array([0, -0.5, 0, 0, -0.5, 0]),
#                                             high=np.array([1, 0.5, 1, 1, 0.5, 1]), dtype=np.float32)

#         self._load_env()

#     def _load_env(self):
#         p.resetSimulation()
#         self.plane_id = p.loadURDF("plane.urdf")
#         self.robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
#         self.end_effector_index = 6
#         self._load_gripper()
#         self._spawn_cube()
#         self._reset_arm()

#     def _load_gripper(self):
#         # Load simple gripper
#         gripper_path = os.path.join(pybullet_data.getDataPath(), "gripper/wsg50_one_motor_gripper.urdf")
#         self.gripper = p.loadURDF(gripper_path, [0, 0, 0], useFixedBase=True)

#     def _spawn_cube(self):
#         self.cube_start_pos = np.array([np.random.uniform(0.2, 0.35), np.random.uniform(-0.1, 0.1), 0.02])
#         self.cube_id = p.loadURDF("cube_small.urdf", self.cube_start_pos.tolist())

#     def _reset_arm(self):
#         for i in range(p.getNumJoints(self.robot)):
#             p.resetJointState(self.robot, i, 0)
#         self.has_grasped = False
#         self.step_counter = 0

#     def reset(self):
#         self._load_env()
#         return self._get_obs()

#     def _get_obs(self):
#         gripper_pos = np.array(p.getLinkState(self.robot, self.end_effector_index)[0])
#         cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
#         return np.concatenate([gripper_pos, cube_pos]).astype(np.float32)

#     def _move_arm_to(self, pos):
#         joint_poses = p.calculateInverseKinematics(self.robot, self.end_effector_index, pos)
#         for i in range(p.getNumJoints(self.robot)):
#             p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, joint_poses[i])
#         for _ in range(20):
#             p.stepSimulation()
#             if self.render:
#                 time.sleep(self.time_step)

#     def _control_gripper(self, open=True):
#         # Placeholder: Add real gripper control here
#         pass

#     def step(self, action):
#         self.step_counter += 1
#         action = np.clip(action, self.action_space.low, self.action_space.high)

#         current_pos = np.array(p.getLinkState(self.robot, self.end_effector_index)[0])
#         new_pos = current_pos + action
#         new_pos = np.clip(new_pos, [0.1, -0.2, 0.02], [0.4, 0.2, 0.3])
#         self._move_arm_to(new_pos)

#         cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
#         xy_dist = np.linalg.norm(new_pos[:2] - cube_pos[:2])

#         reward = -xy_dist
#         done = False

#         if not self.has_grasped and xy_dist < 0.03:
#             self._move_arm_to([cube_pos[0], cube_pos[1], cube_pos[2] + 0.01])
#             self._control_gripper(open=False)
#             time.sleep(0.1)
#             self._move_arm_to([cube_pos[0], cube_pos[1], 0.2])
#             time.sleep(0.1)
#             contacts = p.getContactPoints(bodyA=self.robot, bodyB=self.cube_id)
#             if len(contacts) > 0:
#                 reward += 10.0
#                 self.has_grasped = True
#                 done = True

#         if cube_pos[2] > 0.1:
#             reward += 5.0

#         if self.step_counter >= self.max_steps:
#             done = True

#         return self._get_obs(), reward, done, {}

#     def close(self):
#         p.disconnect(self.physics_client)


# import gym
# import pybullet as p
# import pybullet_data
# import numpy as np
# import time
# from gym import spaces
# import math
# import os

# class SO100PickEnv(gym.Env):
#     def __init__(self, render_mode=False):
#         super(SO100PickEnv, self).__init__()
#         self.render_mode = render_mode
#         self.time_step = 1.0 / 240.0
#         self.max_steps = 200
#         self.step_counter = 0
#         self.urdf_path = "D:/robotics/so100.urdf"

#         self._connect()

#         # Action: delta x, y, z for gripper
#         self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)

#         # Observation: gripper_pos (3), cube_pos (3)
#         obs_low = np.array([-1, -1, 0, -1, -1, 0])
#         obs_high = np.array([1, 1, 1, 1, 1, 1])
#         self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

#     def _connect(self):
#         if self.render_mode:
#             p.connect(p.GUI)
#         else:
#             p.connect(p.DIRECT)
#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         p.setGravity(0, 0, -9.81)

#     def reset(self):
#         p.resetSimulation()
#         p.setGravity(0, 0, -9.81)

#         self.plane = p.loadURDF("plane.urdf")
#         self.robot = p.loadURDF(self.urdf_path, useFixedBase=True)

#         self.joint_indices = [0, 1, 2, 3]
#         self.gripper_left = 4
#         self.gripper_right = 5
#         self.end_effector_index = 3

#         # Random cube position
#         x = np.random.uniform(0.20, 0.30)
#         y = np.random.uniform(-0.05, 0.05)
#         self.cube_pos = [x, y, 0.02]
#         self.cube_id = p.loadURDF("cube_small.urdf", basePosition=self.cube_pos)
#         p.changeDynamics(self.cube_id, -1, lateralFriction=1.0, rollingFriction=0.01)

#         # Open gripper and move to hover
#         self._control_gripper(open=True)
#         self._move_arm_to([x, y, 0.1])

#         self.step_counter = 0
#         self.has_grasped = False
#         return self._get_obs()

#     def _get_obs(self):
#         gripper_pos = np.array(p.getLinkState(self.robot, self.end_effector_index)[0])
#         cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
        
#         obs = np.concatenate([gripper_pos, cube_pos])
#         return obs.astype(np.float32)


#     # def step(self, action):
#     #     self.step_counter += 1

#     #     action = np.clip(action, -0.05, 0.05)
#     #     current_pos = np.array(p.getLinkState(self.robot, self.end_effector_index)[0])
#     #     new_pos = current_pos + action
#     #     new_pos = np.clip(new_pos, [0.1, -0.2, 0.02], [0.4, 0.2, 0.3])
#     #     self._move_arm_to(new_pos)

#     #     cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
#     #     xy_dist = np.linalg.norm(new_pos[:2] - cube_pos[:2])

#     #     reward = -xy_dist
#     #     done = False

#     #     if not self.has_grasped and xy_dist < 0.03:
#     #         # Try to grasp
#     #         self._move_arm_to([cube_pos[0], cube_pos[1], cube_pos[2] + 0.01])
#     #         self._control_gripper(open=False)
#     #         time.sleep(0.2)
#     #         self._move_arm_to([cube_pos[0], cube_pos[1], 0.2])
#     #         time.sleep(0.2)

#     #         # Check contact
#     #         contacts = p.getContactPoints(bodyA=self.robot, bodyB=self.cube_id)
#     #         if len(contacts) > 0:
#     #             reward += 10.0
#     #             self.has_grasped = True
#     #             done = True

#     #     if self.step_counter >= self.max_steps:
#     #         done = True

#     #     return self._get_obs(), reward, done, {}

#     def step(self, action):
#         self.step_counter += 1

#         # Clip the action to  movement per step
#         action = np.clip(action, -0.05, 0.05)

#         # Get current end-effector position
#         current_pos = np.array(p.getLinkState(self.robot, self.end_effector_index)[0])
#         new_pos = current_pos + action
#         new_pos = np.clip(new_pos, [0.1, -0.2, 0.02], [0.4, 0.2, 0.3])
#         self._move_arm_to(new_pos)

#         # Get cube position
#         cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
#         xy_dist = np.linalg.norm(new_pos[:2] - cube_pos[:2])
#         z_dist = abs(new_pos[2] - cube_pos[2])

#         reward = -xy_dist  # penalize distance in XY
#         done = False

#         if not self.has_grasped and xy_dist < 0.03 and z_dist < 0.05:
#             # Try to grasp
#             self._move_arm_to([cube_pos[0], cube_pos[1], cube_pos[2] + 0.005])
#             self._control_gripper(open=False)
#             time.sleep(0.2)

#             # Attempt to lift
#             self._move_arm_to([cube_pos[0], cube_pos[1], 0.2])
#             time.sleep(0.2)

#             # Check if cube was lifted
#             lifted_z = p.getBasePositionAndOrientation(self.cube_id)[0][2]
#             if lifted_z > 0.05:  # cube lifted from table
#                 reward += 10.0
#                 self.has_grasped = True
#                 done = True

#         # Episode end condition
#         if self.step_counter >= self.max_steps:
#             done = True

#         return self._get_obs(), reward, done, {}

#     def _move_arm_to(self, position):
#         orn = p.getQuaternionFromEuler([0, math.pi, 0])
#         joint_angles = p.calculateInverseKinematics(self.robot, self.end_effector_index, position, orn)
#         for i, j in enumerate(self.joint_indices):
#             p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, joint_angles[i])
#         for _ in range(30):
#             p.stepSimulation()
#             if self.render_mode:
#                 time.sleep(self.time_step)

#     # def _control_gripper(self, open=True):
#     #     val = 0.04 if open else -0.01
#     #     force = 10 if open else 200
#     #     p.setJointMotorControl2(self.robot, self.gripper_left, p.POSITION_CONTROL, targetPosition=val, force=force)
#     #     p.setJointMotorControl2(self.robot, self.gripper_right, p.POSITION_CONTROL, targetPosition=-val, force=force)
#     #     for _ in range(20):
#     #         p.stepSimulation()
#     #         if self.render_mode:
#     #             time.sleep(self.time_step)

#     def _control_gripper(self, open=True):
#         target = 0.04 if open else 0.0  # Adjust if your gripper closes tighter
#         p.setJointMotorControl2(self.robot, self.gripper_left, p.POSITION_CONTROL, targetPosition=target, force=100)
#         p.setJointMotorControl2(self.robot, self.gripper_right, p.POSITION_CONTROL, targetPosition=target, force=100)
#         for _ in range(10):  # allow some time for fingers to move
#             p.stepSimulation()
#             if self.render_mode:
#                 time.sleep(self.time_step)

#     def render(self, mode='human'):
#         pass

#     def close(self):
#         p.disconnect()




import gym
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from gym import spaces
import gymnasium as gym 

class PandaPickEnv(gym.Env):
    def __init__(self, render_mode=False):
        super(PandaPickEnv, self).__init__()
        self.render_mode = render_mode
        self.time_step = 1.0 / 240.0
        self.max_steps = 100
        self.step_counter = 0

        self._connect()

        self.arm_joint_indices = list(range(7))
        self.gripper_indices = [9, 10]
        self.ee_link_index = 11

        # Action: delta x, y, z + gripper open/close
        self.action_space = spaces.Box(low=np.array([-0.05, -0.05, -0.05, 0]),
                                       high=np.array([0.05, 0.05, 0.05, 1]),
                                       dtype=np.float32)

        # Observation: gripper pos + cube pos
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def _connect(self):
        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

        x = np.random.uniform(0.3, 0.6)
        y = np.random.uniform(-0.2, 0.2)
        self.cube_pos = [x, y, 0.02]
        self.cube_id = p.loadURDF("cube_small.urdf", basePosition=self.cube_pos)

        self._control_gripper(open=True)
        self._move_arm_to([0.4, 0, 0.3])

        self.step_counter = 0
        self.has_grasped = False
        return self._get_obs()

    def _get_obs(self):
        ee_pos = np.array(p.getLinkState(self.robot, self.ee_link_index)[0])
        cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
        return np.concatenate([ee_pos, cube_pos]).astype(np.float32)

    def step(self, action):
        self.step_counter += 1

        delta = action[:3]
        grip_signal = action[3]

        current_pos = np.array(p.getLinkState(self.robot, self.ee_link_index)[0])
        new_pos = current_pos + delta
        new_pos = np.clip(new_pos, [0.2, -0.3, 0.02], [0.7, 0.3, 0.5])

        self._move_arm_to(new_pos)
        self._control_gripper(open=grip_signal > 0.5)

        cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id)[0])
        dist = np.linalg.norm(new_pos - cube_pos)
        reward = -dist

        done = False

        if not self.has_grasped and dist < 0.05:
            self._control_gripper(open=False)
            time.sleep(0.2)
            self._move_arm_to([new_pos[0], new_pos[1], new_pos[2] + 0.1])
            time.sleep(0.2)
            lifted_z = p.getBasePositionAndOrientation(self.cube_id)[0][2]
            if lifted_z > 0.05:
                reward += 20.0
                self.has_grasped = True
                done = True

        if self.step_counter >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def _move_arm_to(self, position):
        orn = p.getQuaternionFromEuler([0, math.pi, 0])
        joint_angles = p.calculateInverseKinematics(self.robot, self.ee_link_index, position, orn)
        for i, j in enumerate(self.arm_joint_indices):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, joint_angles[i], force=100)
        for _ in range(20):
            p.stepSimulation()
            if self.render_mode:
                time.sleep(self.time_step)

    def _control_gripper(self, open=True):
        target = 0.04 if open else 0.0
        for j in self.gripper_indices:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=target, force=100)
        for _ in range(10):
            p.stepSimulation()
            if self.render_mode:
                time.sleep(self.time_step)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
