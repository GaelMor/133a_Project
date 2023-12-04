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

        self.shoulder_lock = True

        #construct dictionary for indices of joints
        self.indices = self.jointindices()
        self.num_joints = len(self.jointnames())

        # Set up the kinematic chain object.
        self.fl_toe_chain = KinematicChain(node, 'world_yaw_d', 'front_left_toe_link', self.get_jointnames("fl_toe"))
        self.fr_toe_chain = KinematicChain(node, 'world_yaw_d', 'front_right_toe_link', self.get_jointnames("fr_toe"))
        self.rl_toe_chain = KinematicChain(node, 'world_yaw_d', 'rear_left_toe_link', self.get_jointnames("rl_toe"))
        self.rr_toe_chain = KinematicChain(node, 'world_yaw_d', 'rear_right_toe_link', self.get_jointnames("rr_toe"))

        # Define the various points.
        q0 = np.zeros((self.num_joints,1))

        q0[self.indices['base_vert'],0] = 0 #robot's vertical displacement
        q0[self.indices['base_roll'],0] = np.radians(0) #pitch (about y axis)

        q0[self.indices['world_yaw'],0] = 0
        q0[self.indices['world_horiz'],0] = -3
        q0[self.indices['world_vert'],0] = 0.4
        


        toe_q0 = np.array([0 for i in range(len(self.get_jointnames("fl_toe")))]).reshape((-1,1))
        (fl_toe_pos, _, _, _) = self.fl_toe_chain.fkin(toe_q0)
        (fr_toe_pos, _, _, _) = self.fr_toe_chain.fkin(toe_q0)
        (rl_toe_pos, _, _, _) = self.rl_toe_chain.fkin(toe_q0)
        (rr_toe_pos, _, _, _) = self.rr_toe_chain.fkin(toe_q0)
        #self.p0 = [fl_toe_pos, fr_toe_pos, rl_toe_pos, rr_toe_pos]
        self.R0 = Reye()

        # Initialize the current/starting joint position.
        self.qlast  = q0
        self.fl_p0 = fl_toe_pos #desired position when all joints are 0
        self.rl_p0 = rl_toe_pos #desired position when all joints are 0
        self.fr_p0 = fr_toe_pos #desired position when all joints are 0
        self.rr_p0 = rr_toe_pos #desired position when all joints are 0

        self.vec = self.rl_p0 - self.fl_p0 #vector between toes on board
        self.vec_off_board = self.rr_p0 - self.fr_p0 #vector between toes on board

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

        fr_toe_q0 = []
        indexes = self.get_jointindexes("fr_toe")
        for i in indexes:
            fr_toe_q0.append(self.qlast[i,0])
        fr_toe_q0 = np.array(fr_toe_q0).reshape(-1,1)

        rr_toe_q0 = []
        indexes = self.get_jointindexes("rr_toe")
        for i in indexes:
            rr_toe_q0.append(self.qlast[i,0])
        rr_toe_q0 = np.array(rr_toe_q0).reshape(-1,1)

        (self.fl_xd_last, _ , _ , _ ) = self.fl_toe_chain.fkin(fl_toe_q0)
        (self.rl_xd_last, _ , _ , _ )  = self.rl_toe_chain.fkin(rl_toe_q0)
        (self.fr_xd_last, _ , _ , _ ) = self.fr_toe_chain.fkin(fr_toe_q0)
        (self.rr_xd_last, _ , _ , _ )  = self.rr_toe_chain.fkin(rr_toe_q0)
        self.Rd_last = self.R0
        self.lam = 20

        #self.pub = node.create_publisher(Float64, '/condition', 10)   
        self.toe_delta = np.array([0.06, 0, 0.04]).reshape(-1,1)

        self.l_gamma = 0.01 #gamma used for left legs (the ones on board)
        self.r_gamma = 0.02 #gamma used for right legs (the ones off the board)

        self.minz = 100 #used for debugging

        self.board_height = 0.016
        self.offset = self.board_height + 0.004

        self.dir = np.array([1,0,0]).reshape(-1,1) #initial direction that world is moving
        self.curr_dir = self.dir
        self.prev_world_horiz = 0
        self.prev_world_vert = 0

        self.d_theta = -pi/2 # desired theta for secondary task (elbow up vs elbow down)
        self.lam_ls = 10 #secondary lambda value for left toes
        self.lam_rs = 10 #secondary lambda value for right toes



    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!        
        return ['world_horiz','world_vert', 'world_y', 'world_yaw', 'base_vert', 'base_roll', 'base_pitch', 
                'front_left_shoulder', 'front_left_leg', 'front_left_foot', 
                'front_right_shoulder', 'front_right_leg', 'front_right_foot', 
                'rear_left_shoulder', 'rear_left_leg', 'rear_left_foot', 
                'rear_right_shoulder', 'rear_right_leg', 'rear_right_foot', 'attach-board']

    def jointindices(self):
        jointnames = self.jointnames()
        indices = {}
        for i in range(len(jointnames)):
            indices[jointnames[i]] = i
        return indices

    def get_jointindexes(self, name):
        joints = self.get_jointnames(name)
        indices = []
        for joint in joints:
            indices.append(self.indices[joint])
        return indices

    def get_jointnames(self, name):
        fl_toe_chain_joints = ['base_vert', 'base_roll', 'base_pitch', 'front_left_shoulder', 'front_left_leg', 'front_left_foot']
        fr_toe_chain_joints = ['base_vert', 'base_roll', 'base_pitch', 'front_right_shoulder', 'front_right_leg', 'front_right_foot']
        rl_toe_chain_joints = ['base_vert', 'base_roll', 'base_pitch', 'rear_left_shoulder', 'rear_left_leg', 'rear_left_foot']
        rr_toe_chain_joints = ['base_vert', 'base_roll', 'base_pitch', 'rear_right_shoulder', 'rear_right_leg', 'rear_right_foot'] 


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
        self.qlast[self.indices['world_yaw'],0] = np.radians(10) * sin(t)

        self.curr_dir = Rotz(self.qlast[self.indices['world_yaw'],0]) @ Roty(self.qlast[self.indices['base_roll'],0]) @ self.dir

        self.qlast[self.indices['world_horiz'],0] += (self.curr_dir[0,0] * 0.005)  #makes robot move in x direction
        if self.qlast[self.indices['world_horiz'],0] >= 3:
            self.qlast[self.indices['world_horiz'],0] = -3

        self.qlast[self.indices['world_y'],0] += (self.curr_dir[1,0] * 0.005)
        self.qlast[self.indices['world_vert'],0] += (self.curr_dir[2,0] * 0.005) #makes robot move in z direction

        self.qlast[self.indices['base_vert'],0] = -0.1 * sin(0.4*t) * sin(0.4*t)
        pitch_omega = 1.0
        roll_omega = 0.50
        self.qlast[self.indices['base_roll'],0] = np.radians(15) * sin(pitch_omega *t) #pitch (about y axis)
        self.qlast[self.indices['base_pitch'],0] = np.radians(15) * sin(roll_omega *t) #roll (about x axis)


        # desired positions for left toes
        fl_pd = self.fl_p0 
        fl_vd = np.array([0,0,0]).reshape(-1,1)

        rl_pd = self.fl_p0  + Roty(self.qlast[self.indices['base_roll'],0]) @ self.vec 
        rl_vd = np.array([0,0,0]).reshape(-1,1)
        rl_vd[0,0] = -1 * self.vec[0,0] * sin(np.radians(15) * sin(pitch_omega*t)) * np.radians(15) * cos(pitch_omega*t) * pitch_omega  + self.vec[2,0] * cos(np.radians(15) * sin(pitch_omega*t)) * np.radians(15) * cos(pitch_omega  * t) * pitch_omega 
        rl_vd[1,0] = 0 
        rl_vd[2,0] = -1 * self.vec[0,0] * cos(np.radians(15) * sin(pitch_omega*t)) * np.radians(15) * cos(pitch_omega*t) * pitch_omega  - self.vec[2,0] * sin(np.radians(15) * sin(pitch_omega*t)) * np.radians(15) * cos(pitch_omega*t) * pitch_omega 

        # desired positions for right toes
        fr_omega = 3.0
        fr_pd = np.zeros((3,1))
        fr_pd[0,0] =  self.fr_p0[0,0] - self.toe_delta[0,0] * sin(fr_omega * t)
        fr_pd[1,0] =  self.fr_p0[1,0] 
        fr_pd[2,0] =  (self.fr_p0[2,0] + self.toe_delta[2,0]) - self.toe_delta[2,0] * cos(fr_omega * t) - self.offset
        fr_pd = self.fr_p0 + Roty(self.qlast[self.indices['base_roll'],0]) @ (fr_pd - self.fr_p0)

        fr_vd = np.zeros((3,1))
        fr_vd[0,0] =  -1 * self.toe_delta[0,0] * cos(fr_omega * t) * fr_omega
        fr_vd[1,0] =  0
        fr_vd[2,0] =  -2 * self.toe_delta[2,0] * sin(fr_omega * t) * fr_omega
        fr_vd = Roty(self.qlast[self.indices['base_roll'],0]) @ fr_vd


        rr_omega = 3.0
        rr_pd = fr_pd  + Roty(self.qlast[self.indices['base_roll'],0]) @ self.vec_off_board

        rr_vd = np.zeros((3,1))
        rr_vd[0,0] =  -1 * self.toe_delta[0,0] * cos(rr_omega * t) * rr_omega
        rr_vd[1,0] =  0
        rr_vd[2,0] =  -2 * self.toe_delta[2,0] * sin(rr_omega * t) * rr_omega
        rr_vd = Roty(self.qlast[self.indices['base_roll'],0]) @ rr_vd


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


        (fl_p, _ , fl_Jv, _ ) = self.fl_toe_chain.fkin(fl_qlast)
        (rl_p, _ , rl_Jv, _ ) = self.rl_toe_chain.fkin(rl_qlast)
        (fr_p, fr_R, fr_Jv, fr_Jw) = self.fr_toe_chain.fkin(fr_qlast)
        (rr_p, rr_R, rr_Jv, rr_Jw) = self.rr_toe_chain.fkin(rr_qlast)
        
        qdot = np.zeros((self.num_joints,1))

        # inverse kinematics for fl toe
        # Compute the errors
        error_pos = ep(self.fl_xd_last, fl_p)
        error = error_pos

        # compute qdot
        v = fl_vd
        A = v + self.lam * error
    
        fl_Jv[:,0:3] = 0
        if self.shoulder_lock:
            fl_Jv[:,0:4] = 0

        J = fl_Jv
        JT = np.transpose(J)
        JW_pinv = JT @ np.linalg.inv(J @ JT + (self.l_gamma**2) * np.eye(3)) 
        qdot_s = np.zeros((6,1))
        qdot_s[5,0] = self.lam_ls * (-pi/2 - self.qlast[self.indices['front_left_foot'], 0])
        fl_qdot = JW_pinv @ A + ((np.eye(6) - JW_pinv @ J) @ qdot_s)
        #fl_qdot = JW_pinv @ A 
        #fl_qdot = np.linalg.pinv(J) @ A

        # Integrate the joint position.
        indexes = self.get_jointindexes("fl_toe")
        j  = 0
        for i in indexes:
            if i > 6:
                qdot[i,0] = fl_qdot[j,0]
            j += 1


        # inverse kinematics for rl toe
        # Compute the errors
        error_pos = ep(self.rl_xd_last, rl_p)
        error = error_pos

        # compute qdot
        v = rl_vd
        A = v + self.lam * error
    
        rl_Jv[:,0:3] = 0
        if self.shoulder_lock:
            rl_Jv[:,0:4] = 0
        J = rl_Jv

        JT = np.transpose(J)
        JW_pinv = JT @ np.linalg.inv(J @ JT + (self.l_gamma**2) * np.eye(3)) 
        qdot_s = np.zeros((6,1))
        qdot_s[5,0] = self.lam_ls * (-pi/2 - self.qlast[self.indices['rear_left_foot'], 0])
        rl_qdot = JW_pinv @ A + ((np.eye(6) - JW_pinv @ J) @ qdot_s)
        #rl_qdot = JW_pinv @ A

        # Integrate the joint position.
        indexes = self.get_jointindexes("rl_toe")
        j  = 0
        for i in indexes:
            if i > 6:
                qdot[i,0] = rl_qdot[j,0]
            j += 1


        ######
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
        fr_Jv[:,0:3] = 0
        J = fr_Jv

        JT = np.transpose(J)
        JW_pinv = JT @ np.linalg.inv(J @ JT + (self.r_gamma**2) * np.eye(3)) 
        qdot_s = np.zeros((6,1))
        qdot_s[5,0] = self.lam_ls * (-pi/2 - self.qlast[self.indices['front_right_foot'], 0])
        fr_qdot = JW_pinv @ A + ((np.eye(6) - JW_pinv @ J) @ qdot_s)

        #fr_qdot = JW_pinv @ A

        # Integrate the joint position.
        indexes = self.get_jointindexes("fr_toe")
        j  = 0
        for i in indexes:
            if i > 6:
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
        rr_Jv[:,0:3] = 0
        J = rr_Jv

        JT = np.transpose(J)
        JW_pinv = JT @ np.linalg.inv(J @ JT + (self.r_gamma**2) * np.eye(3)) 
        qdot_s = np.zeros((6,1))
        qdot_s[5,0] = self.lam_ls * (-pi/2 - self.qlast[self.indices['rear_right_foot'], 0])
        rr_qdot = JW_pinv @ A + ((np.eye(6) - JW_pinv @ J) @ qdot_s)
        #rr_qdot = JW_pinv @ A

        # Integrate the joint position.
        indexes = self.get_jointindexes("rr_toe")
        j = 0
        for i in indexes:
            if i > 6:
                qdot[i,0] = rr_qdot[j,0]
            j += 1
        

        q = self.qlast + dt * qdot
        q[self.indices['attach-board'],0] = -1 *(q[self.indices['front_left_leg'],0] + q[self.indices['front_left_foot'],0])
        #print(q)


        # Save the data needed next cycle.
        self.qlast = q
        self.fl_xd_last = fl_pd
        self.rl_xd_last = rl_pd
        self.fr_xd_last = fr_pd
        self.rr_xd_last = rr_pd
        #self.fl_Rd_last = fl_Rd
        #self.rl_Rd_last = rl_Rd
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
