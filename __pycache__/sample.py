import pybullet as p
import pybullet_data
import time

# Connect to PyBullet GUI
p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load plane and robot
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("D:/robotics/so100.urdf", basePosition=[0, 0, 0])

# Load cube object
cube_start_pos = [0.6, 0, 0.05]
cube = p.loadURDF("cube_small.urdf", basePosition=cube_start_pos)

# Step a bit to let things settle
for _ in range(100):
    p.stepSimulation()
    time.sleep(1./240.)

# Identify end-effector (KUKA's is link 6)
end_effector_index = 6

# === STEP 1: Move arm above the cube ===
approach_pos = [0.6, 0, 0.3]  # approach from above
joint_angles = p.calculateInverseKinematics(robot, end_effector_index, approach_pos)
for i in range(len(joint_angles)):
    p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, joint_angles[i])

for _ in range(100):
    p.stepSimulation()
    time.sleep(1./240.)

# === STEP 2: Move arm down to the cube ===
pickup_pos = [0.6, 0, 0.05]  # closer to cube
joint_angles = p.calculateInverseKinematics(robot, end_effector_index, pickup_pos)
for i in range(len(joint_angles)):
    p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, joint_angles[i])

for _ in range(200):
    p.stepSimulation()
    time.sleep(1./240.)

# === STEP 3: Simulate gripping the cube (create constraint) ===
constraint_id = p.createConstraint(
    parentBodyUniqueId=robot,
    parentLinkIndex=end_effector_index,
    childBodyUniqueId=cube,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0]
)

# === STEP 4: Lift the cube ===
lift_pos = [0.6, 0, 0.3]  # lift straight up
joint_angles = p.calculateInverseKinematics(robot, end_effector_index, lift_pos)
for i in range(len(joint_angles)):
    p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, joint_angles[i])

for _ in range(200):
    p.stepSimulation()
    time.sleep(1./240.)

# === STEP 5: Move to another location ===
place_pos = [0.4, 0.3, 0.3]
joint_angles = p.calculateInverseKinematics(robot, end_effector_index, place_pos)
for i in range(len(joint_angles)):
    p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, joint_angles[i])

for _ in range(200):
    p.stepSimulation()
    time.sleep(1./240.)

print("Pick and place completed.")
time.sleep(5)
p.disconnect()
