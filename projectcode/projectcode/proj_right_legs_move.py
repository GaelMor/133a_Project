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

        toe_q0 = np.array([0 for i in range(len(self.get_jointnames("fl_toe")))]).reshape((-1,1))
        (fl_toe_pos, _, _, _) = self.fl_toe_chain.fkin(toe_q0)
        (fr_toe_pos, _, _, _) = self.fr_toe_chain.fkin(toe_q0)
        (rl_toe_pos, _, _, _) = self.rl_toe_chain.fkin(toe_q0)
        (rr_toe_pos, _, _, _) = self.rr_toe_chain.fkin(toe_q0)
        #self.p0 = [fl_toe_pos, fr_toe_pos, rl_toe_pos, rr_toe_pos]
        self.R0 = Reye()

        # Initialize the current/starting joint position.
        self.qlast  = self.q0

        self.fr_p0 = fr_toe_pos
        self.rr_p0 = rr_toe_pos
        self.fr_xd_last = fr_toe_pos
        self.rr_xd_last = rr_toe_pos
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
        elif name == "rl_name":
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

        # desired positions for right toes
        fr_pd = self.fr_p0 + self.toe_delta * sin(5*t) * sin(5*t)
        fr_vd = 2 * self.toe_delta * sin(5*t) * cos(5*t) * 5

        rr_pd = self.rr_p0 + self.toe_delta * sin(5*t) * sin(5*t)
        rr_vd = 2 * self.toe_delta * sin(5*t) * cos(5*t) * 5


        # kinematic chains
        fr_qlast = []
        indexes = self.get_jointindexes("fr_toe")
        for i in indexes:
            fr_qlast.append(self.qlast[i,0])
        fr_qlast = np.array(fr_qlast).reshape(-1,1)


        rr_qlast = []
        indexes = self.get_jointindexes("rr_toe")
        for i in indexes:
            rr_qlast.append(self.qlast[i,0])
        rr_qlast = np.array(rr_qlast).reshape(-1,1)


        (fr_p, fr_R, fr_Jv, fr_Jw) = self.fr_toe_chain.fkin(fr_qlast)
        (rr_p, rr_R, rr_Jv, rr_Jw) = self.rr_toe_chain.fkin(rr_qlast)
        qdot = np.radians(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((-1,1)))

        # inverse kinematics for fr toe
        # Compute the errors
        error_pos = ep(self.fr_xd_last, fr_p)
        #error = np.vstack((error_pos, error_rot))
        error = error_pos

        # compute qdot
        #v = np.vstack((vd,wd))
        v = fr_vd
        A = v + self.lam * error
        #J = np.vstack((Jv, Jw))
        fr_Jv[:,0:4] = 0
        J = fr_Jv
        fr_qdot = np.linalg.pinv(J) @ A

        # Integrate the joint position.
        indexes = self.get_jointindexes("fr_toe")
        j  = 0
        for i in indexes:
            if i > 3:
                qdot[i,0] = fr_qdot[j,0]
            j += 1


        # inverse kinematics for rr toe
        # Compute the errors
        error_pos = ep(self.rr_xd_last, rr_p)
        #error = np.vstack((error_pos, error_rot))
        error = error_pos

        # compute qdot
        #v = np.vstack((vd,wd))
        v = rr_vd
        A = v + self.lam * error
        #J = np.vstack((Jv, Jw))
        rr_Jv[:,0:4] = 0
        J = rr_Jv
        rr_qdot = np.linalg.pinv(J) @ A

        # Integrate the joint position.
        indexes = self.get_jointindexes("rr_toe")
        j = 0
        for i in indexes:
            if i > 3:
                qdot[i,0] = rr_qdot[j,0]
            j += 1


        q = self.qlast + dt * qdot

        # Save the data needed next cycle.
        self.qlast = q
        self.fr_xd_last = fr_pd
        self.rr_xd_last = rr_pd
        #self.Rd_last = Rd
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
