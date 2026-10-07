import Franka
import numpy as np

# Initialize robot object
mybot = Franka.FrankArm()

# Compute forward kinematics
deg_to_rad = np.pi/180.

joint_targets = [[0., 0., 0., 0., 0., 0., 0.],
                 [0, 0, -45.*deg_to_rad, -15.*deg_to_rad, 20. *
                     deg_to_rad, 15.*deg_to_rad, -75.*deg_to_rad],
                 [0, 0, 30.*deg_to_rad, -60.*deg_to_rad, -65. *
                     deg_to_rad, 45.*deg_to_rad, 0.*deg_to_rad],
                 [0.17145329,  0.15406996, -0.22092292, -2.08190828, 
                  -0.03134244,  3.75481774, -2.89957154],
                 [0.1806371793818485, 0.2573696168181983, -0.20968635151721726, -2.082814327211186, 
                  -0.03294013587439925, 3.9070531278949088, -3.0644869188862067]
                 ]

for joint_target in joint_targets:
    print('\nJoints:')
    print(joint_target)
    Hcurr, J = mybot.ForwardKin(joint_target)
    ee_pose = Hcurr[-1]
    rot_ee = ee_pose[:3, :3]
    pos_ee = ee_pose[:3, 3]
    print('computed FK ee position')
    print(pos_ee)
    print('computed FK ee rotation')
    print(rot_ee)


# Compute inverse kinematics
qInit = [0, 0, 0, -2.11, 0, 3.65, -0.785]
HGoal = np.array([[0., 0., 1., 0.6],  # target EE pose
                 [0., 1., 0., 0.0],
                 [-1., 0, 0., 0.5],
                 [0., 0., 0., 1]])

q, Err = mybot.IterInvKin(qInit, HGoal)
print('Error', np.linalg.norm(Err[0:3]), np.linalg.norm(Err[3:6]))
print('Computed IK angles', q)
