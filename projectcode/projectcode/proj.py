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
        fl_toe_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'front_left_shoulder', 'front_left_leg', 'front_left_foot']
        fr_toe_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'front_right_shoulder', 'front_right_leg', 'front_right_foot']
        rl_toe_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'rear_left_shoulder', 'rear_left_leg', 'rear_left_foot']
        rr_toe_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'rear_right_shoulder', 'rear_right_leg', 'rear_right_foot'] 
        self.fl_toe_chain = KinematicChain(node, 'world', 'front_left_toe_link', fl_toe_chain_joints)
        self.fr_toe_chain = KinematicChain(node, 'world', 'front_right_toe_link', fr_toe_chain_joints)
        self.rl_toe_chain = KinematicChain(node, 'world', 'rear_left_toe_link', rl_toe_chain_joints)
        self.rr_toe_chain = KinematicChain(node, 'world', 'rear_right_toe_link', rr_toe_chain_joints)

        fl_shoulder_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'front_left_shoulder']
        self.fl_shoulder_chain = KinematicChain(node, 'world', 'front_left_shoulder_link', fl_shoulder_chain_joints)

        fl_foot_chain_joints = ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 'front_left_shoulder', 'front_left_leg', 'front_left_foot']
        self.fl_foot_chain = KinematicChain(node, 'world', 'front_left_foot_link', fl_foot_chain_joints)

        # Define the various points.
        self.q0 = np.radians(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((-1,1)))
        self.p0 = np.array([0.0, 0.55, 1.0]).reshape((-1,1))
        self.R0 = Reye()

        self.pleft  = np.array([0.3, 0.5, 0.15]).reshape((-1,1))
        self.phigh = np.array([0.0, 0.5, 0.9]).reshape((-1,1))
        self.pright = np.array([-0.3, 0.5, 0.15]).reshape((-1,1))

        # Initialize the current/starting joint position.
        self.qlast  = self.q0
        self.xd_last = self.p0
        self.Rd_last = self.R0
        self.lam = 20

        #self.pub = node.create_publisher(Float64, '/condition', 10)


    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!        
        return ['base_horiz', 'base_vert', 'base_roll', 'base_pitch', 
                'front_left_shoulder', 'front_left_leg', 'front_left_foot', 
                'front_right_shoulder', 'front_right_leg', 'front_right_foot', 
                'rear_left_shoulder', 'rear_left_leg', 'rear_left_foot', 
                'rear_right_shoulder', 'rear_right_leg', 'rear_right_foot']

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        (pshoulder_l, _, _, _) = self.fl_shoulder_chain.fkin(self.qlast[0:5,:])
        (pfoot_l, _, _, _) = self.fl_foot_chain.fkin(self.qlast[0:7,:])
        (ptoe_l, _, _, _) = self.fl_toe_chain.fkin(self.qlast[0:7,:])
        #print("Left Shoulder's position: {}".format(pshoulder_l))
        #print("Left Foot's position: {}".format(pfoot_l))
        #print("Left Toe's position: {}".format(ptoe_l))

        q = self.q0
        qdot = self.q0
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
