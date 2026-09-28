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
# Functions:		tcpposecb, jointstatecb, armstatuscb, switch_controller, publish_zero_twist, process_waypoints, main
# Nodes:		     Publishing Topics  - [ /delta_twist_cmds, /delta_joint_cmds ]
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

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Add any variable your motion needs to keep between cycles.
        #       ->  HINT: Which waypoint you are on, and what the arm is doing about it.

        ############################################

        # Variable to store target waypoints
        self.target_waypoints = waypoints
        #waypoint index for current waypoint 
        self.current_wp_idx = 0

        #waypoint arrival and hold state
        self.is_holding = False 
        self.hold_start_time = None
        self.hold_duration = 2.0
        self.distance_tolerance = 0.020 #20mm tolernaace allowed

        #proportional controler gains 
        self.kp_linear = 0.9 if self.current_wp_idx == 1 else 1.2
        self.kp_angular = 1.0
        #orientation lock for the wrist
        self.target_orientation = None 

        #Controller switch management 
        self.controller_active = False 
        self.switch_future = None 


    def tcpposecb(self, data):
        '''
        Description:    Callback function for the tool pose topic.
                        Use this function to receive where the tool currently is.

        Args:
            data (PoseStamped):    Pose of the tool, reported in base_link

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Store the tool's position and orientation.
        #       ->  HINT: data.pose.position, data.pose.orientation

        ############################################
        self.tcp_pose = data.pose

        #Set reference orientation (after first callback)
        if self.target_orientation is None and self.tcp_pose:
            self.target_orientation = self.tcp_pose.orientation


    def jointstatecb(self, data):
        '''
        Description:    Callback function for the joint states topic.
                        Use this function to receive the current angle of each joint.

        Args:
            data (JointState):    Joint feedback published by the arm

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Store the joint angles, matched BY NAME - the order is not promised.
        #       ->  HINT: angles = [data.position[data.name.index(j)] for j in joint_names]
        #       ->  NOTE: A joint assumed to be at zero when it is not reads as a large
        #                 error, and a controller acting on it drives hard towards it.

        ############################################
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

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Store the code, and log it while you are developing. Zero is healthy; anything
        #       else is the arm telling you about the last command or its own state.
        #       ->  HINT: /arm_status_detail says the same thing in words.
        #       ->  NOTE: A protective stop LATCHES and cannot be cleared - the run is over.

        ############################################
        self.arm_status = data.data


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

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Call the switch_controller service on self.switch_cli-
        #           req = SwitchController.Request()
        #           req.activate_controllers = [ the one you want ]
        #           req.deactivate_controllers = [ the other one ]
        #           req.strictness = SwitchController.Request.STRICT

        #   ->  Wait for it first, and expect the first attempt to fail-
        #           self.switch_cli.wait_for_service(timeout_sec=20.0)
        #           future = self.switch_cli.call_async(req)
        #           rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        #       ->  NOTE: That last call HANGS if made from a callback of a node that is
        #                 already spinning. __init__ is safe; from the timer, poll
        #                 'future.done()' instead.

        #   ->  Do not switch more often than you need to.

        ############################################
        req = SwitchController.Request()
        req.activate_controllers = [controller]
        req.deactivate_controllers = [joint_controller if controller == twist_controller else twist_controller]
        req.strictness = SwitchController.Request.STRICT
        
        if not self.switch_cli.service_is_ready():
            self.get_logger().error("switch controller is not ready")
            return False

        self.switch_future = self.switch_cli.call_async(req)
        return True

    def publish_zero_twist(self):
        """Publishes zero velocity command to prevent coasting and satisfy command timeout."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        self.twist_pub.publish(msg)

    def process_waypoints(self):
        '''
        Description:    Timer function used to drive the tool through the waypoints.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Return early until pose, joints and status have all arrived.

        #   ->  Both topics carry VELOCITIES, never positions.

        #   ->  Build the message for whichever interface you made active:
        #           TwistStamped on self.twist_pub  - linear and angular velocity, in 'base_frame'
        #           JointJog on self.joint_pub      - 'joint_names' and a velocity for each
        #       ->  HINT: msg.header.stamp = self.get_clock().now().to_msg()      (both)
        #                 msg.header.frame_id = base_frame                        (twist)
        #                 msg.twist.linear.x/.y/.z, msg.twist.angular.x/.y/.z     (twist)
        #                 msg.joint_names = joint_names, msg.velocities = [...]   (jog)

        #   ->  Publish on EVERY tick, zero included - see 'command_timeout_s'. Never sleep
        #       inside this function.

        #   ->  Drive the tool at a velocity proportional to the error-
        #           v = Kp * (target - current)
        #       ->  HINT: Cap it by scaling the WHOLE vector: v = v * (cap / |v|)
        #       ->  NOTE: The servo ramps down rather than stopping dead, so the arm settles
        #                 PAST the pose that satisfied you.

        #   ->  Command the tool's ORIENTATION too, or the servo decides the wrist for you.
        #       ->  HINT: The turn from unit vector 'a' onto unit vector 'b', as an axis
        #                 times an angle-
        #                     c     = cross(a, b)
        #                     e     = c / |c| * atan2(|c|, dot(a, b))
        #                     omega = Kp * e
        #                 'a' is the tool's own axis - a column of
        #                 Rotation.from_quat([x, y, z, w]).as_matrix().

        #   ->  Think about the PATH. The arm has joint limits and configurations it cannot
        #       pass through.

        #   ->  Track which waypoint you are on, and log the distance to it while developing.

        ############################################
        # Heartbeat check : Return early and publish zero twist until all data has arrived 
        if self.tcp_pose is None or self.joint_angles is None or self.arm_status is None:
            self.publish_zero_twist()
            return
        #check if all waypoints are completed 
        if self.current_wp_idx >= len(self.target_waypoints):
            self.publish_zero_twist()
            return
        
        # extract position and target
        current_pos = np.array([self.tcp_pose.position.x, self.tcp_pose.position.y, self.tcp_pose.position.z])
        target_pos = np.array(self.target_waypoints[self.current_wp_idx])

        # position error vector and distance 
        error_pos = target_pos - current_pos 
        distance = np.linalg.norm(error_pos)

        #logging distance while developing
        self.get_logger().info(
            f"WP {self.current_wp_idx + 1}/{len(self.target_waypoints)} | Distance: {distance:.4f} m", 
            throttle_duration_sec=0.5
        )

        now = self.get_clock().now()

        #waypoint state machiene and holding logic 
        if distance <= self.distance_tolerance:
            if not self.is_holding:
                self.is_holding = True 
                self.hold_start_time = now
                self.get_logger().info(f"Reached waypoint {self.current_wp_idx + 1} holding for {self.hold_duration} seconds")

            elasped = (now - self.hold_start_time).nanoseconds / 1e9 # convert to seconds 

            if elasped >= self.hold_duration:
                self.is_holding = False 
                self.current_wp_idx += 1
                self.get_logger().info(f"Finished holding at waypoint {self.current_wp_idx + 1}")
            self.publish_zero_twist()
            return 

        # reset holding flag if moved out of tolerance 
        self.is_holding = False

        #calculate the linear velocity component using a PD controller 
        Kp_lin = 1.2
        v_lin = Kp_lin * error_pos 
        v_mag = np.linalg.norm(v_lin)

        #Scale whole vector using cap_linear_mps with 2% safety margin (0.147 m/s)
        max_cap = cap_linear_mps * 0.98

        if v_mag > max_cap: 
            v_lin = v_lin * (max_cap/v_mag)

        #Calculate the angular velocity using Tool Axis alignment logic 
        # a_tcp is the tool's own z axis 
        q = [self.tcp_pose.orientation.x, self.tcp_pose.orientation.y, self.tcp_pose.orientation.z, self.tcp_pose.orientation.w]

        R_tcp = Rotation.from_quat(q).as_matrix()

        # Extract tool axes from the rotation matrix 
        a_tcp = R_tcp[:, 2] # Z axis 

        # b_tcp is the target tool vector (aligning tool axis along direction of motion to prevent wrist locks)
        b_tcp = error_pos / distance if distance > 1e-4 else a_tcp

        #Alignment cross product and dot product 
        c = np.cross(a_tcp, b_tcp)
        c_norm = np.linalg.norm(c)
        dot_a_tcp_b_tcp = np.dot(a_tcp, b_tcp)

        kp_ang = 0.8
        if c_norm > 1e-6:
            e = (c / c_norm) * math.atan2(c_norm, dot_a_tcp_b_tcp)
            omega = kp_ang * e
        else:
            #If vectors are aligned (near zero cross product)
            omega = np.zeros(3)
        
        # Cap angular velocity magnitude for smoothness
        omega_mag = np.linalg.norm(omega)
        max_ang_cap = 0.5  # rad/s
        if omega_mag > max_ang_cap:
            omega = omega * (max_ang_cap / omega_mag)

        #Construct and publish TwistStamped message
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = base_frame
        twist_msg.twist.linear.x = v_lin[0]
        twist_msg.twist.linear.y = v_lin[1]
        twist_msg.twist.linear.z = v_lin[2]
        twist_msg.twist.angular.x = omega[0]
        twist_msg.twist.angular.y = omega[1]
        twist_msg.twist.angular.z = omega[2]

        self.twist_pub.publish(twist_msg)
    
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
