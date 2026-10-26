#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 2B of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          1454
# Author List:		Ayush Tiwari, Aryan Joshi
# Filename:		    path_follower.py
# Functions:        pathcb, odomcb, scancb, lookup_laser_offset, get_base_frame_obstacles,
#                   get_closest_obstacle_in_sector, measure_side_clearances, normalize_angle,
#                   send_velocity, stop_ebot, process_navigation, main
# Nodes:		    Publishing Topics  - [ /cmd_vel ]
#                   Subscribing Topics - [ /ebot_path, /odom, /scan ]

################### IMPORT MODULES #######################

import rclpy
import sys
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
import tf2_ros

from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
) # import the QoSProfile and related policies to set the subscription to /ebot_path correctly

from sensor_msgs.msg import LaserScan




##################### TASK CONSTANTS #######################

# The route is published here ONCE, before your node starts, and held for late
# subscribers. An ordinary subscription waits forever for a message already sent.
path_topic = '/ebot_path'

odom_topic = '/odom'
scan_topic = '/scan'
cmd_topic = '/cmd_vel'

# What /cmd_vel accepts. The base clips anything beyond these, and clipping one of the
# two changes their ratio - which is the arc the base actually drives.
max_linear_mps = 0.5
max_angular_rps = 1.0

# The subscription to /ebot_path has to match the publisher's QoS, or it will never receive the route.

path_qos = QoSProfile(
    depth=1,
    history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

CRUISE_SPEED = 0.40 # m/s, under the 0.5 m/s cap
LOOKAHEAD = 0.50            # m, how far ahead along the path the eBot aims


# Pose-controller gains
LINEAR_KP = 0.8
ANGULAR_KP = 2
MIN_TRACK_SPEED = 0.05      # m/s, so it never stalls just short of the end

# The path is done when the eBot is this close to its last point, then it turns on the spot to the yaw the path ends with. The task guide allows 0.3 m and 0.15 rad.

FINAL_POSITION_TOL = 0.08   # m
FINAL_YAW_TOL = 0.10        # rad
YAW_KP = 1.5

ROTATE_MAX = 0.4          # rad/s
ROTATE_MIN = 0.15           # rad/s, below this a skid-steer base does not turn at all
RESUME_DISTANCE = 0.25      # m, if the turn pushes it this far off the last point, drive back






def wrap_angle(angle):
    '''
    Description:    Wrap an angle to [-pi, pi], so an error is never read the long way round.
 
    Args:
        angle (float):          Angle in radians
 
    Returns:
        wrapped_angle (float):  Angle wrapped to [-pi, pi]
    '''
    return math.atan2(math.sin(angle), math.cos(angle))

class path_tracker:
    '''
    ___CLASS___
 
    Description: The following logic, with no ROS in it: given where the eBot is, say how to drive. It runs through TRACK (follow the path), ROTATE (turn to the end yaw) and HOLD (stand still until the next path).
    '''
 
    def __init__(self):
        '''
        Description:    Initialization of class path_tracker
        '''
        self.points = []            # the path, [(x, y), ...]
        self.final_yaw = 0.0        # the yaw the path ends with
        self.index = 0              # how far along the path the eBot has got
        self.phase = 'IDLE'         # IDLE, TRACK, ROTATE or HOLD
 
    def set_path(self, points, final_yaw):
        '''
        Description:    Start following a new path.
 
        Args:
            points (list):      [(x, y), ...]
            final_yaw (float):  the yaw to finish on, in radians
 
        Returns:
        '''
        self.points = list(points) #list of points to follow
        self.final_yaw = final_yaw #final yaw to reach
        self.index = 0 #index of the current point
        self.phase = 'TRACK' #initial phase
 
    def step(self, x, y, yaw):
        '''
        Description:    One control cycle.
 
        Args:
            x, y (float):   where the eBot is, in the map frame
            yaw (float):    which way it faces, in radians
 
        Returns:
            (linear, angular) (float, float):   the velocities to send, in m/s and rad/s
        '''
        #if ebot is idle or holding, return 0 velocity
        if self.phase in ('IDLE', 'HOLD'):
            return 0.0, 0.0
 
        last_x, last_y = self.points[-1] #last point in the path
        to_end = math.hypot(last_x - x, last_y - y) #distance from ebot to last point
 
        #if ebot is close to the end point, switch to rotate phase
        if self.phase == 'ROTATE':
            error = wrap_angle(self.final_yaw - yaw)
            if abs(error) <= FINAL_YAW_TOL:
                self.phase = 'HOLD'
                return 0.0, 0.0
            rate = max(ROTATE_MIN, min(ROTATE_MAX, YAW_KP * abs(error)))
            return 0.0, math.copysign(rate, error)
 
        # TRACK: move the progress marker on while the next point is nearer than this one
        count = len(self.points)
        while (self.index + 1 < count and
               math.hypot(self.points[self.index + 1][0] - x, self.points[self.index + 1][1] - y)
               <= math.hypot(self.points[self.index][0] - x, self.points[self.index][1] - y)):
            self.index += 1
 
        #if ebot is close to the end point and close to the final yaw, switch to rotate phase
        if to_end <= FINAL_POSITION_TOL and self.index >= count - 2:
            self.phase = 'ROTATE'
            return 0.0, 0.0
 
        # TRACK: aim at the first point at least LOOKAHEAD away, or at the end of the path
        target = self.index

        while target < count - 1 and math.hypot(self.points[target][0] - x, self.points[target][1] - y) < LOOKAHEAD:
            target += 1
        tx, ty = self.points[target]
 
        heading_error = wrap_angle(math.atan2(ty - y, tx - x) - yaw)

        # 0 when the target is behind: turn first
        facing = max(0.0, math.cos(heading_error))          
 
        linear = min(CRUISE_SPEED, LINEAR_KP * to_end) * facing

        # to avoid stalling just short of the end
        if facing > 0.3:
            linear = max(linear, MIN_TRACK_SPEED)

        # angular velocity is proportional to the heading error
        angular = ANGULAR_KP * heading_error

        return linear, angular
 
 
##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the eBot along the route
                    published on /ebot_path, through its waypoints, in order.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_nav
        '''

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
        super().__init__(                                                               # registering node
            'ebot_nav_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic SUBSCRIPTIONS ############

        # ->  NOTE: The subscription to 'path_topic' below uses the default settings and will
        #           NEVER receive the route - see the INSTRUCTIONS in 'pathcb' and fix it there.
        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, path_qos)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)



        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)                      # the eBot drives on what you publish here




        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05 # rate of time to run one control cycle (seconds)

        self.timer = self.create_timer(control_rate, self.process_navigation) # creating a timer based function which gets called on every 0.05 seconds (as defined by 'control_rate' variable)


        self.tracker = path_tracker() # the following logic

        self.last_phase = 'IDLE' # to log each change of phase once
        self.odom = None


    def pathcb(self, data):
        '''
        Description:    Callback function for the route topic.
                        Use this function to receive the path for the step about to be driven.
 
        Args:
            data (Path):    The path, as a sequence of poses
 
        Returns:
        '''
        # store path points in the tracker
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in data.poses]

        # if path has less than 2 points, ignore it
        if len(points) < 2:
            self.get_logger().warn("Ignoring a path with fewer than two poses")
            return
 
        # The last pose says which way to face at the end of the step.
        orientation_quaternion = data.poses[-1].pose.orientation

        # convert quaternion to yaw
        final_yaw = math.atan2(2.0 * (orientation_quaternion.w * orientation_quaternion.z + orientation_quaternion.x * orientation_quaternion.y), 1.0 - 2.0 * (orientation_quaternion.y * orientation_quaternion.y + orientation_quaternion.z * orientation_quaternion.z))
 
        # Unlike Task 1C, every new path is taken: the planner sends one per step.
        self.tracker.set_path(points, final_yaw)

        # logging path info to console
        self.get_logger().info(
            f"Path received: {len(points)} poses in frame '{data.header.frame_id}', "
            f"ending at ({points[-1][0]:.2f}, {points[-1][1]:.2f}) facing {final_yaw:.2f} rad")


    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.
                        Use this function to receive where the base currently is.

        Args:
            data (Odometry):    Pose and velocity of the base

        Returns:
        '''

        base_pos = data.pose.pose.position # position of the base
        base_quat = data.pose.pose.orientation # orientation of the base (quaternion)

        # Convert quaternion to yaw angle
        yaw = math.atan2(2.0 * (base_quat.w * base_quat.z + base_quat.x * base_quat.y), 1.0 - 2.0 * (base_quat.y * base_quat.y + base_quat.z * base_quat.z))

        self.odom = (base_pos.x, base_pos.y, yaw)


    def send_velocity(self, linear, angular):
        '''
        Description:    Publish a Twist message to the cmd_vel topic, with linear and angular
                        velocities clipped to the maximums defined above.

        Args:
            linear (float):     Linear velocity in m/s
            angular (float):    Angular velocity in rad/s 

        Returns:
        '''
        msg = Twist() # create a new Twist message

        # clipping the linear and angular velocities to the maximums defined above and publishing the message to the cmd_vel topic
        msg.linear.x = float(max(-max_linear_mps, min(max_linear_mps, linear)))
        
        msg.angular.z = float(max(-max_angular_rps, min(max_angular_rps, angular)))

        self.cmd_pub.publish(msg) # publish the Twist message to the cmd_vel topic


    def stop_ebot(self):
        '''
        Description:    Publish zero linear and angular velocity to stop the eBot.

        Args:
        Returns:
        '''
        self.send_velocity(0.0, 0.0) # publish a Twist message with zero linear and angular velocities to stop the robot


    def process_navigation(self):
        '''
        Description:    Timer function used to drive the eBot along the path.
 
        Args:
        Returns:
        '''
 
        # Until the first position arrives, stand still
        if self.odom is None:
            self.stop_ebot()
            return
 
        # an uncaught exception kills the node and leaves the base coasting
        try:
            odom_x, odom_y, odom_yaw = self.odom
            linear, angular = self.tracker.step(odom_x, odom_y, odom_yaw)
            self.send_velocity(linear, angular)
 
            if self.tracker.phase != self.last_phase:
                self.last_phase = self.tracker.phase
                self.get_logger().info(f"Following: {self.last_phase} at x={odom_x:.2f} y={odom_y:.2f} yaw={odom_yaw:.2f}")
 
        except Exception as nav_error:
            self.get_logger().error(f"process_navigation failed: {nav_error}")
            self.stop_ebot()





##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_nav class to perform its task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    ebot_nav_class = ebot_nav()                                     # creating a new object for class 'ebot_nav'
    ebot_nav_class.get_logger().info('Node created: eBot navigation process')

    try:
        rclpy.spin(ebot_nav_class)                                  # spinning on the object to make it alive in ROS 2 DDS
    except KeyboardInterrupt:
        pass
    finally:
        ebot_nav_class.destroy_node()                               # destroy node after spin ends
        if rclpy.ok():
            rclpy.shutdown()                                        # shutdown process


if __name__ == '__main__':

    main()
