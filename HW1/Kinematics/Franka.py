import numpy as np
import RobotUtil as rt
import math

class FrankArm:
    def __init__(self):
        # Robot descriptor taken from URDF file (rpy xyz for each rigid link transform) - NOTE: don't change
        self.Rdesc = [
            [0, 0, 0, 0., 0, 0.333],  # From robot base to joint1
            [-np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0, -0.316, 0],
            [np.pi/2, 0, 0, 0.0825, 0, 0],
            [-np.pi/2, 0, 0, -0.0825, 0.384, 0],
            [np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0.088, 0, 0],
            [0, 0, 0, 0, 0, 0.107]  # From joint5 to end-effector center
        ]

        # Define the axis of rotation for each joint
        self.axis = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]

        # Set base coordinate frame as identity - NOTE: don't change
        self.Tbase = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

        # Initialize matrices - NOTE: don't change this part
        self.Tlink = []  # Transforms for each link (const)
        self.Tjoint = []  # Transforms for each joint (init eye)
        self.Tcurr = []  # Coordinate frame of current (init eye)
        
        for i in range(len(self.Rdesc)):
            self.Tlink.append(rt.rpyxyz2H(
                self.Rdesc[i][0:3], self.Rdesc[i][3:6]))
            self.Tcurr.append([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 1, 0.], [0, 0, 0, 1]])
            self.Tjoint.append([[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 1, 0.], [0, 0, 0, 1]])

        self.Tlinkzero = rt.rpyxyz2H(self.Rdesc[0][0:3], self.Rdesc[0][3:6])

        self.Tlink[0] = np.matmul(self.Tbase, self.Tlink[0])

        # initialize Jacobian matrix
        self.J = np.zeros((6, 7))

        self.q = [0., 0., 0., 0., 0., 0., 0.]
        self.q2 = np.deg2rad([0, 0, -45, -15, 20, 15, -75])
        self.q2 = np.deg2rad([0, 0, 30, -60, -65, 45, 0])

    def ForwardKin(self, ang):
        '''
        inputs: joint angles
        outputs: joint transforms for each joint, Jacobian matrix
        '''

        self.q[0:-1] = ang

        # Compute current joint and end effector coordinate frames (self.Tjoint). Remember that not all joints rotate about the z axis!
        for i in range(7):
            self.Tjoint[i] = rt.MatrixExp(self.axis[i], self.q[i])

        self.Tjoint[7] = np.eye(4)

        self.Tcurr[0] = self.Tlink[0] @ self.Tjoint[0]
        for i in range(1, 8):
            self.Tcurr[i] = self.Tcurr[i-1] @ self.Tlink[i] @ self.Tjoint[i]

        p_ee = self.Tcurr[-1][:3, 3]
        for i in range(7):
            p_i = self.Tcurr[i][:3, 3]
            z_i = self.Tcurr[i][:3, :3] @ np.asarray(self.axis[i])

            self.J[:3, i] = np.cross(z_i, p_ee - p_i)
            self.J[3:, i] = z_i

        return self.Tcurr, self.J

    def IterInvKin(self, ang, TGoal, x_eps=1e-3, r_eps=1e-3):
        '''
        inputs: starting joint angles (ang), target end effector pose (TGoal)

        outputs: computed joint angles to achieve desired end effector pose, 
        Error in your IK solution compared to the desired target
        '''

        q = np.asarray(ang, dtype=float).copy()
        Tcurr, J = self.ForwardKin(q)
        position_error = TGoal[:3, 3] - Tcurr[-1][:3, 3]
        rotation_error = TGoal[:3, :3] @ Tcurr[-1][:3, :3].T
        axis, ang = rt.R2axisang(rotation_error)
        rotation_error = np.array(axis) * ang
        Err = np.concatenate((position_error, rotation_error))


        C = np.diag([1_000_000, 1_000_000, 1_000_000,
             1_000, 1_000, 1_000])
        W = np.diag([1, 1, 100, 100, 1, 1, 100])


        count = 0
        while (np.linalg.norm(position_error) > x_eps or np.linalg.norm(rotation_error) > r_eps):
            if count >= 1000: break
            A = J.T @ C @ J + W
            b = J.T @ C @ Err
            delta_q = np.linalg.solve(A, b)
            q = q + delta_q
            Tcurr, J = self.ForwardKin(q)
            position_error = TGoal[:3, 3] - Tcurr[-1][:3, 3]
            rotation_error = TGoal[:3, :3] @ Tcurr[-1][:3, :3].T
            axis, ang = rt.R2axisang(rotation_error)
            rotation_error = np.array(axis) * ang
            Err = np.concatenate((position_error, rotation_error))
            count += 1

        return self.q[0:-1], Err
