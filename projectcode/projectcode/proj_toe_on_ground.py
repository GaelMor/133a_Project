'''
proj.py
'''

import rclpy
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from hw5code.KinematicChain     import KinematicChain
from std_msgs.msg import Float64


#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.fl_toe_chain = KinematicChain(node, 'world', 'front_left_toe_link', self.get_jointnames("fl_toe"))
        self.fr_toe_chain = KinematicChain(node, 'world', 'front_right_toe_link', self.get_jointnames("fr_toe"))
        self.rl_toe_chain = KinematicChain(node, 'world', 'rear_left_toe_link', self.get_jointnames("rl_toe"))
        self.rr_toe_chain = KinematicChain(node, 'world', 'rear_right_toe_link', self.get_jointnames("rr_toe"))

        # Define the various points.
        self.q0 = np.radians(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((-1,1)))
        self.q0[1,0] = -0.1

        toe_q0 = np.array([0 for i in range(len(self.get_jointnames("fl_toe")))]).reshape((-1,1))
        (fl_toe_pos, _, _, _) = self.fl_toe_chain.fkin(toe_q0)
        (fr_toe_pos, _, _, _) = self.fr_toe_chain.fkin(toe_q0)
        (rl_toe_pos, _, _, _) = self.rl_toe_chain.fkin(toe_q0)
        (rr_toe_pos, _, _, _) = self.rr_toe_chain.fkin(toe_q0)
        #self.p0 = [fl_toe_pos, fr_toe_pos, rl_toe_pos, rr_toe_pos]
        self.R0 = Reye()

        # Initialize the current/starting joint position.
        self.qlast  = self.q0
        self.fl_p0 = fl_toe_pos #desired position
        self.rl_p0 = rl_toe_pos #desired position

        # actual initial positions stored in xd last
        fl_toe_q0 = []
        indexes = self.get_jointindexes("fl_toe")
        for i in indexes:
            fl_toe_q0.append(self.qlast[i,0])
        fl_toe_q0 = np.array(fl_toe_q0).reshape(-1,1)

        rl_toe_q0 = []
        indexes = self.get_jointindexes("rl_toe")
        for i in indexes:
            rl_toe_q0.append(self.qlast[i,0])
        rl_toe_q0 = np.array(rl_toe_q0).reshape(-1,1)

        (self.fl_xd_last, self.fl_Rd_last, _ , _ ) = self.fl_toe_chain.fkin(fl_toe_q0)
        (self.rl_xd_last, self.rl_Rd_last, _ , _ )  = self.rl_toe_chain.fkin(rl_toe_q0)
        self.Rd_last = self.R0
        self.lam = 20

        #self.pub = node.create_publisher(Float64, '/condition', 10)

        self.toe_delta = np.array([0.1, 0, 0.1]).reshape(-1,1)


    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!        
        return ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 
                'front_left_shoulder', 'front_left_leg', 'front_left_foot', 
                'front_right_shoulder', 'front_right_leg', 'front_right_foot', 
                'rear_left_shoulder', 'rear_left_leg', 'rear_left_foot', 
                'rear_right_shoulder', 'rear_right_leg', 'rear_right_foot']

    def get_jointindexes(self, name):
        if name == "fl_toe":
            return [0,1,2,3,4,5,6]
        elif name == "fr_toe":
            return [0,1,2,3,7,8,9]
        elif name == "rl_toe":
            return [0,1,2,3,10,11,12]
        else:
            return [0,1,2,3,13,14,15]

    def get_jointnames(self, name):
        fl_toe_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'front_left_shoulder', 'front_left_leg', 'front_left_foot']
        fr_toe_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'front_right_shoulder', 'front_right_leg', 'front_right_foot']
        rl_toe_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'rear_left_shoulder', 'rear_left_leg', 'rear_left_foot']
        rr_toe_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'rear_right_shoulder', 'rear_right_leg', 'rear_right_foot'] 

        fl_shoulder_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'front_left_shoulder']
        fr_shoulder_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'front_right_shoulder']
        rl_shoulder_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'rear_left_shoulder']
        rr_shoulder_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'rear_right_shoulder']

        if name == "fl_toe":
            return fl_toe_chain_joints
        elif name == "fr_toe":
            return fr_toe_chain_joints 
        elif name == "rl_toe":
            return rl_toe_chain_joints
        elif name == "rr_toe":
            return rr_toe_chain_joints

        elif name == "fl_shoulder":
            return fl_shoulder_chain_joints
        elif name == "fr_shoulder":
            return fr_shoulder_chain_joints
        elif name == "rl_shoulder":
            return rl_shoulder_chain_joints
        else:
            return rr_shoulder_chain_joints

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):

        self.qlast[1,0] = -0.1 * sin(0.1*t) * sin(0.1*t)

        # desired positions for right toes
        fl_pd = self.fl_p0 
        fl_vd = np.array([0,0,0]).reshape(-1,1)
        fl_Rd = Reye()
        fl_wd = np.array([0,0,0]).reshape(-1,1)

        rl_pd = self.rl_p0 
        rl_vd = np.array([0,0,0]).reshape(-1,1)
        rl_Rd = Reye()
        rl_wd = np.array([0,0,0]).reshape(-1,1)


        # kinematic chains
        fl_qlast = []
        indexes = self.get_jointindexes("fl_toe")
        for i in indexes:
            fl_qlast.append(self.qlast[i,0])
        fl_qlast = np.array(fl_qlast).reshape(-1,1)


        rl_qlast = []
        indexes = self.get_jointindexes("rl_toe")
        for i in indexes:
            rl_qlast.append(self.qlast[i,0])
        rl_qlast = np.array(rl_qlast).reshape(-1,1)


        (fl_p, fl_R, fl_Jv, fl_Jw) = self.fl_toe_chain.fkin(fl_qlast)
        (rl_p, rl_R, rl_Jv, rl_Jw) = self.rl_toe_chain.fkin(rl_qlast)
        qdot = np.radians(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((-1,1)))

        # inverse kinematics for fl toe
        # Compute the errors
        error_pos = ep(self.fl_xd_last, fl_p)
        error_rot = eR(self.fl_Rd_last, fl_R)
        error = np.vstack((error_pos, error_rot))

        # compute qdot
        vd = fl_vd
        wd = fl_wd
        v = np.vstack((vd,wd))
        A = v + self.lam * error
    
        fl_Jv[:,0:4] = 0
        fl_Jw[:,0:4] = 0
        J = np.vstack((fl_Jv, fl_Jw))
        fl_qdot = np.linalg.pinv(J) @ A

        # Integrate the joint position.
        indexes = self.get_jointindexes("fl_toe")
        j  = 0
        for i in indexes:
            if i > 3:
                qdot[i,0] = fl_qdot[j,0]
            j += 1


        # inverse kinematics for rl toe
        # Compute the errors
        error_pos = ep(self.rl_xd_last, rl_p)
        error_rot = eR(self.rl_Rd_last, rl_R)
        error = np.vstack((error_pos, error_rot))

        # compute qdot
        vd = rl_vd
        wd = rl_wd
        v = np.vstack((vd,wd))
        A = v + self.lam * error
    
        rl_Jv[:,0:4] = 0
        rl_Jw[:,0:4] = 0
        J = np.vstack((rl_Jv, rl_Jw))
        rl_qdot = np.linalg.pinv(J) @ A

        # Integrate the joint position.
        indexes = self.get_jointindexes("rl_toe")
        j  = 0
        for i in indexes:
            if i > 3:
                qdot[i,0] = rl_qdot[j,0]
            j += 1


        q = self.qlast + dt * qdot
        print(q)


        # Save the data needed next cycle.
        self.qlast = q
        self.fl_xd_last = fl_pd
        self.rl_xd_last = rl_pd
        self.fl_Rd_last = fl_Rd
        self.rl_Rd_last = rl_Rd
        return (q.flatten().tolist(), qdot.flatten().tolist())


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
