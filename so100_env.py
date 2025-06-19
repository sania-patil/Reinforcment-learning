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
