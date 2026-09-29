#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1B of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          1454
# Author List:		Aryan Joshi
# Filename:		    arm_waypoints.py
# Functions:		main
# Nodes:		    Publishing Topics  - [ /delta_twist_cmds, /delta_joint_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw, /joint_states, /arm_status ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
from rclpy.node import Node
from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from scipy.spatial.transform import Rotation
import numpy as np

##################### TASK CONSTANTS #######################

# Tool positions in base_link, in metres, in the order they must be reached. The tool
# stops at each one and holds it for at least two seconds. Copy the signs as they are:
# base_link is the UR7e's own frame, not the Gazebo world's.
waypoints = [
    (-0.4085, -0.5379, 0.1967),   # 1
    (-0.8000, -0.0005, 0.3967),   # 2
    (-0.7430,  0.5280, 0.1967),   # 3
    (-0.4097,  0.5280, 0.1967),   # 4
    (-0.0763,  0.5280, 0.1967),   # 5
]

# The two command interfaces. Only ONE is active at a time; messages to the other are
# accepted and ignored.
servo_ns = '/ur_arm_controller'
twist_controller = 'delta_twist_controller'
joint_controller = 'delta_joint_controller'

# The only frame a twist may be stamped with; any other is refused, not converted. Note
# 'base' is base_link turned through 180 degrees, not another name for it.
base_frame = 'base_link'

# JointJog velocities are matched to these names, in this order.
joint_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# What the servo accepts. A command above one of these is dropped WHOLE, not clamped.
cap_linear_mps = 0.15     # magnitude of a twist's linear part
cap_angular_rps = 0.35    # magnitude of its angular part
cap_joint_rps = 0.35      # per joint

# Dead-man switch: the arm stops this long after the last message it received.
command_timeout_s = 0.15

# A joint-space pose with the elbow mid-range, used to move off the singular
# start pose before Cartesian tracking begins.
unfold_targets = {
    'shoulder_pan_joint':  0.0,
    'shoulder_lift_joint': -1.20,
    'elbow_joint':         -1.60,
    'wrist_1_joint':       -1.90,
    'wrist_2_joint':        1.57,
    'wrist_3_joint':        1.57,
}

unfold_tolerance_rad = 0.05 # tolerance for the joint angles

kp_joint = 0.8 # proportional gains for the joint angles


##################### CLASS DEFINITION #######################

class arm_waypoints(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the UR7e's tool through the
                    given waypoints using the arm's velocity command interfaces.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class arm_waypoints
        '''

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
        super().__init__(                                                               # registering node
            'arm_waypoints_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic PUBLISHERS ############

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)    # end-effector velocity, in base_link
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)        # per-joint velocity

        ############ Topic SUBSCRIPTIONS ############

        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05                                                             # rate of time to run one control cycle (seconds)
        self.switch_cli = self.create_client(                                           # client used to pick which command topic is live
            SwitchController, f'{servo_ns}/switch_controller')
        self.timer = self.create_timer(control_rate, self.process_waypoints)            # creating a timer based function which gets called on every 0.05 seconds (as defined by 'control_rate' variable)

        self.tcp_pose = None                                                            # tool pose variable (from tcpposecb())
        self.joint_angles = None                                                        # joint feedback variable (from jointstatecb())
        self.arm_status = None                                                          # arm state code variable (from armstatuscb())

        # waypoint targets
        self.target_waypoints = waypoints  # Variable to store target waypoints
        self.current_wp_idx = 0 #waypoint index for current waypoint

        #waypoint arrival and hold state variables
        self.is_holding = False             # flag to check if the arm is holding the waypoint
        self.hold_start_time = None         #timer for the hold duration
        self.hold_duration = 2.0            #duration for which the arm holds the waypoint
        self.distance_tolerance = 0.020     #20mm tolernaace allowed

        #proportional controler gains 
        self.kp_linear = 1.2                # proportional gains for the linear velocity 
        self.kp_angular = 0.2               # proportional gains for the angular velocity
        
        self.target_orientation = None      #orientation lock for the wrist

        #Controller switch management 
        self.active_controller = None       # active controller state variables
        self.switch_future = None           #switch future state
        self.switch_pending = None          #switch pending state 

        self.phase = 'unfold'               # phase state variables

    def tcpposecb(self, data):
        '''
        Description:    Callback function for the tool pose topic.
                        Use this function to receive where the tool currently is.

        Args:
            data (PoseStamped):    Pose of the tool, reported in base_link

        Returns:
        '''

        #store tool position and orientation
        self.tcp_pose = data.pose

    def jointstatecb(self, data):
        '''
        Description:    Callback function for the joint states topic.
                        Use this function to receive the current angle of each joint.

        Args:
            data (JointState):    Joint feedback published by the arm

        Returns:
        '''

        if all(name in data.name for name in joint_names):
            self.joint_angles = [data.position[data.name.index(j)] for j in joint_names]


    def armstatuscb(self, data):
        '''
        Description:    Callback function for the arm status topic.
                        Use this function to receive the arm's current state code.

        Args:
            data (Int32):    One state code describing what the arm is doing

        Returns:
        '''

        self.arm_status = data.data #store arm status


    def switch_controller(self, controller):
        '''
        Description:    Function to make one of the arm's two command interfaces the active
                        one, so that commands published to it are acted on.

        Args:
            controller  (str):      Name of the controller to activate, either
                                    'twist_controller' or 'joint_controller'

        Returns:
            success     (bool):     Whether the controller was activated
        '''

        # activating and deactivating controllers
        req = SwitchController.Request() #create request for controller switching
        req.activate_controllers = [controller]  # activate required controller 
        req.deactivate_controllers = [joint_controller if controller == twist_controller else twist_controller] # deactivate otehr controller 
        req.strictness = SwitchController.Request.STRICT # strict mode of operation

        #check if the service is ready
        if not self.switch_cli.service_is_ready():
            self.get_logger().error("switch controller is not ready")
            return False

        self.switch_future = self.switch_cli.call_async(req)    #call the service asynchronously
        self.switch_pending = controller                        #set the pending controller
        return True

    def publish_zero_twist(self):
        '''
        Description:    Publish zero twist commands to the arm to prevent coasting and satisfy the command timeout.
        Args:
        Returns:
        '''
        
        #create zero twist message
        msg = TwistStamped()                                    #create twist stamped message
        msg.header.stamp = self.get_clock().now().to_msg()      #set the header stamp
        msg.header.frame_id = base_frame                        #set the header frame id 
        msg.twist.linear.x = 0.0                                #set linear velocity
        msg.twist.linear.y = 0.0                                #set linear velocity
        msg.twist.linear.z = 0.0                                #set linear velocity
        msg.twist.angular.x = 0.0                               #set angular velocity
        msg.twist.angular.y = 0.0                               #set angular velocity
        msg.twist.angular.z = 0.0                               #set angular velocity
        self.twist_pub.publish(msg)                             #publish the zero twist message

    def publish_zero_joint(self):
        '''
        Description:    Publish zero joint velocities to the arm to prevent coasting and satisfy the command timeout.
        Args:
        Returns:
        '''

        #create zero joint velocity message
        msg = JointJog()                                        #create joint jog message
        msg.header.stamp = self.get_clock().now().to_msg()      #set the header stamp
        msg.joint_names = joint_names                           #set the joint names
        msg.velocities = [0.0] * len(joint_names)               #set the joint velocities
        self.joint_pub.publish(msg)                             #publish the zero joint velocity message

    def run_unfold(self):
        '''
        Description:    Drive the joints toward unfold_targets using switch_controller.
        Args:
        Returns:
        '''

        #check if the active controller is the joint controller
        if self.active_controller != joint_controller:
            #check if the switch future is None
            if self.switch_future is None:
                self.switch_controller(joint_controller)                #switch to the joint controller
            elif self.switch_future.done():                             #check if the switch future is done
                try:
                    res = self.switch_future.result()
                    if res.ok:
                        self.active_controller = self.switch_pending
                except Exception:
                    pass
                self.switch_future = None
            self.publish_zero_twist()
            return

        #calculate the errors between the current joint angles and the unfold targets
        errors = [unfold_targets[name] - angle
                  for name, angle in zip(joint_names, self.joint_angles)]

        #check if the maximum error is within the tolerance
        if max(abs(e) for e in errors) < unfold_tolerance_rad:
            self.publish_zero_joint()
            self.target_orientation = self.tcp_pose.orientation 
            self.phase = 'track'
            self.get_logger().info("Unfold complete, switching to waypoint tracking")
            return

        # calculate the cap limits for joint velocities
        cap = cap_joint_rps * 0.98 # reduces the cap by 2% to prevent jerk
        velocities = [max(-cap, min(cap, kp_joint * e)) for e in errors] # calculate the joint velocities using PID

        #create joint jog message
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()      #set the header stamp
        msg.joint_names = joint_names                           #set the joint names
        msg.velocities = velocities                             #set the joint velocities
        self.joint_pub.publish(msg)

    def run_tracking(self):
        '''
        Description:    Drive the arm towards the target waypoints using switch_controller.
        Args:
        Returns:
        '''
        
        #check if the active controller is the twist controller
        if self.active_controller != twist_controller:
            #check if the switch future is None
            if self.switch_future is None:
                self.switch_controller(twist_controller)                #switch to the twist controller

            #check if the switch future is done
            elif self.switch_future.done():
                try:
                    res = self.switch_future.result()
                    if res.ok:
                        self.active_controller = self.switch_pending
                        self.get_logger().info("Activated delta_twist_controller successfully.")
                except Exception:
                    pass
                self.switch_future = None
            self.publish_zero_twist()
            return

        #check if all waypoints are reached
        if self.current_wp_idx >= len(self.target_waypoints):
            self.publish_zero_twist() # publish zero twist to stop the arm
            return

        #calculate the error between the current position and the target position
        current_pos = np.array([self.tcp_pose.position.x, self.tcp_pose.position.y, self.tcp_pose.position.z])
        
        target_pos = np.array(self.target_waypoints[self.current_wp_idx]) #get the target position

        error_pos = target_pos - current_pos # error position variable

        #calculate the distance between the current position and the target position
        distance = np.linalg.norm(error_pos) 

        #logging the current waypoint and the distance to the target waypoint
        self.get_logger().info(
            f"WP {self.current_wp_idx + 1}/{len(self.target_waypoints)} | Distance: {distance:.4f} m",
            throttle_duration_sec=0.5
        )

        now = self.get_clock().now() #get the current time 

        #check if the distance is within the tolerance
        if distance <= self.distance_tolerance:
            #check if the arm is not holding
            if not self.is_holding:
                self.is_holding = True
                self.hold_start_time = now
                self.get_logger().info(f"Reached waypoint {self.current_wp_idx + 1} holding for {self.hold_duration} seconds")
            
            #check if the hold duration is reached
            elapsed = (now - self.hold_start_time).nanoseconds / 1e9
            if elapsed >= self.hold_duration:
                self.is_holding = False
                self.current_wp_idx += 1
                self.get_logger().info(f"Finished holding at waypoint {self.current_wp_idx}")
            self.publish_zero_twist()
            return

        self.is_holding = False                 # reset the holding flag

        #check if the current waypoint is the first waypoint
        Kp_lin = 0.9 if self.current_wp_idx == 0 else self.kp_linear
        
        v_lin = Kp_lin * error_pos              #linear velocity
        v_mag = np.linalg.norm(v_lin)           #magnitude of linear velocity
        max_cap = cap_linear_mps * 0.98         #maximum linear velocity

        #cap the linear velocity
        if v_mag > max_cap:
            v_lin = v_lin * (max_cap / v_mag)

        q_curr = [self.tcp_pose.orientation.x, self.tcp_pose.orientation.y, self.tcp_pose.orientation.z, self.tcp_pose.orientation.w]  #quaternion of the current orientation    

        R_curr = Rotation.from_quat(q_curr).as_matrix() # current rotation matrix
        a_curr = R_curr[:, 2]                           # z-axis of the current orientation

        q_target = [self.target_orientation.x, self.target_orientation.y, self.target_orientation.z, self.target_orientation.w] #quaternion of the target orientation

        R_target = Rotation.from_quat(q_target).as_matrix() # target rotation matrix
        b_target = R_target[:, 2] # z-axis of the target orientation

        cross_product = np.cross(a_curr, b_target) #cross product of the current and target z-axes

        c_norm = np.linalg.norm(cross_product) #magnitude of the cross product
        dot_a_b = np.dot(a_curr, b_target) #dot product of the current and target z-axes

        if c_norm > 1e-6:
            error = (cross_product / c_norm) * math.atan2(c_norm, dot_a_b) #angular error
            omega = self.kp_angular * error
        else:
            omega = np.zeros(3)

        #cap the angular velocity
        omega_mag = np.linalg.norm(omega) #magnitude of angular velocity
        max_ang_cap = cap_angular_rps * 0.98 #maximum angular velocity
        if omega_mag > max_ang_cap:
            omega = omega * (max_ang_cap / omega_mag)

        #create the twist message
        twist_msg = TwistStamped() #create a twistStamped message
        twist_msg.header.stamp = self.get_clock().now().to_msg() #header stamp
        twist_msg.header.frame_id = base_frame #frame id

        twist_msg.twist.linear.x = float(v_lin[0]) #linear velocity x-component
        twist_msg.twist.linear.y = float(v_lin[1]) #linear velocity y-component
        twist_msg.twist.linear.z = float(v_lin[2]) #linear velocity z-component

        twist_msg.twist.angular.x = float(omega[0]) #angular velocity x-component
        twist_msg.twist.angular.y = float(omega[1]) #angular velocity y-component
        twist_msg.twist.angular.z = float(omega[2]) #angular velocity z-component

        self.twist_pub.publish(twist_msg) #publish the twist message 
        
    def process_waypoints(self):
        '''
        Description:    Timer function used to drive the tool through the waypoints.

        Args:
        Returns:
        '''

        # Heartbeat check : Return early and publish zero twist until all data has arrived 
        if self.tcp_pose is None or self.joint_angles is None or self.arm_status is None:
            self.publish_zero_twist()
            return
        
        # check the current phase and run the corresponding method
        if self.phase == 'unfold':
            self.run_unfold()
        else:
            self.run_tracking()

    
##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    arm_waypoints class to perform its task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('arm_waypoints_process')               # creating ROS node

    node.get_logger().info('Node created: Arm waypoints process')   # logging information

    arm_waypoints_class = arm_waypoints()                           # creating a new object for class 'arm_waypoints'

    rclpy.spin(arm_waypoints_class)                                 # spining on the object to make it alive in ROS 2 DDS

    arm_waypoints_class.destroy_node()                              # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':

    main()
