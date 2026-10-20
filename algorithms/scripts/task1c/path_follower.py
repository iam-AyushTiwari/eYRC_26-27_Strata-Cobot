#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1C of StrataCobot (SC) Theme (eYRC 2026-27).
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

WAYPOINT_REACHED_DIST = 0.25

# Pose-controller gains
LINEAR_KP = 0.8
ANGULAR_KP = 2

# The obstacle avoidance controller is a simple turn away from the nearest thing 

FRONT_SECTOR_ANGLE = math.radians(35.0)      # Radians of the cone ahead to look for obstacles 
OBSTACLE_TRIGGER_DIST = 0.7                  # Meters to keep from the nearest obstacle
STEER_AVOID_KP = 1.2                         # Radians to turn away from the nearest obstacle, per meter of clearance deficit
AVOID_SPEED_LIMIT = max_linear_mps * 0.6     # MPS to drive when the nearest obstacle is at the safety distance

ROBOT_HALF_WIDTH = 0.30                      # Meters from the centre of the base to its widest part (the wheels)
CLEARANCE_MARGIN = 0.15                      # Meters to keep from the nearest obstacle, on top of the robot's half-width

REQUIRED_SIDE_CLEARANCE = ROBOT_HALF_WIDTH + CLEARANCE_MARGIN  # Meters of minimum passable room

CLEARANCE_DIFF_TOLERANCE = 0.15              # Meters to consider two openings as the same



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
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scancb, qos_profile_sensor_data)

        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)                      # the eBot drives on what you publish here

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05                                                             # rate of time to run one control cycle (seconds)
        self.timer = self.create_timer(control_rate, self.process_navigation)           # creating a timer based function which gets called on every 0.05 seconds (as defined by 'control_rate' variable)

        self.route = None                                                               # the route to drive (from pathcb())
        self.odom = None                                                                # where the base is (from odomcb())
        self.scan = None                                                                # what the lidar sees (from scancb())

        ############################################

        self.current_waypoint_index = 0 # index of the waypoint currently being driven towards
        self.route_completed = False # flag to log the completion of the route only once

        self.tf_buffer = tf2_ros.Buffer() # buffer for storing transform data
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self) # listener for receiving transform data
        self.lidar_tf_ready = False # flag to indicate if the lidar transform is ready
        self.lidar_dx = 0.0 # x offset of the lidar from the base center
        self.lidar_dy = 0.0 # y offset of the lidar from the base center
        self.lidar_yaw_offset = 0.0 # yaw offset of the lidar from the base frame



    def pathcb(self, data):
        '''
        Description:    Callback function for the route topic.
                        Use this function to receive the waypoints the eBot has to drive.

        Args:
            data (Path):    The route, as a sequence of poses

        Returns:
        '''


        if self.route is not None: 
            return  # ignore any new route once one has been received

        # Store the route as a list of (x, y) tuples
        self.route = [(pose.pose.position.x, pose.pose.position.y) for pose in data.poses] 
        self.get_logger().info(
            f"Route received: {len(self.route)} waypoints in frame '{data.header.frame_id}'")


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


    def scancb(self, data):
        '''
        Description:    Callback function for the lidar topic.
                        Use this function to receive what the lidar currently sees.

        Args:
            data (LaserScan):    One lidar sweep

        Returns:
        '''

        self.scan = data # store the entire LaserScan message for later processing in process_navigation()

    def lookup_laser_offset(self):
        '''
        Description:    Look up the lidar's mounting offset from ebot_base_link, once, and
                        cache it. Returns False (and keeps retrying) until the tree has
                        filled in.

        Returns:
            ready (bool):   Whether self.lidar_dx/_dy/_yaw_offset are valid
        '''

        # checking if the lidar transform is already ready, if yes then return True
        if self.lidar_tf_ready:
            return True

        # checking if the scan message is available, if not then return False
        try:
            laser_transform = self.tf_buffer.lookup_transform(
                'ebot_base_link', self.scan.header.frame_id, rclpy.time.Time())
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as tf_error:
            self.get_logger().warn(f"Lidar TF lookup failed: {tf_error}", throttle_duration_sec=2.0)
            return False

        translation = laser_transform.transform.translation # translation
        quaternion = laser_transform.transform.rotation # rotation (quaternion)
        self.lidar_dx = translation.x # x offset of the lidar from the base center
        self.lidar_dy = translation.y # y offset of the lidar from the base center
        
        # calculating the yaw offset of the lidar from the base frame using quaternion to euler conversion
        self.lidar_yaw_offset = math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y), 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z))
        self.lidar_tf_ready = True

        # Log the lidar offset from ebot_base_link, in meters and degrees, with 3 decimal places for dx and dy, and 1 decimal place for yaw
        self.get_logger().info(
            f"Lidar offset from ebot_base_link: dx={self.lidar_dx:.3f} m, "
            f"dy={self.lidar_dy:.3f} m, yaw={math.degrees(self.lidar_yaw_offset):.1f} deg")
        return True

    def get_base_frame_obstacles(self):
        '''
        Description:    Filter the current scan to valid rays, and express each one as a
                        (distance, bearing) pair in ebot_base_link rather than the lidar frame.

        Returns:
            points (list):  (distance, bearing) tuples, bearing in radians, 0 = straight
                            ahead of the base, positive = to the left
        '''
        scan = self.scan # use the latest scan message stored in self.scan
        cos_yaw = math.cos(self.lidar_yaw_offset) # cosine of the yaw offset of the lidar from the base frame
        sin_yaw = math.sin(self.lidar_yaw_offset) # sine of the yaw offset of the lidar from the base frame

        points = []
        # Filter the scan to valid rays and express each one as a (distance, bearing) pair in ebot_base_link
        for beam_idx, beam_range in enumerate(scan.ranges):
            if not math.isfinite(beam_range):
                continue
            if beam_range < scan.range_min or beam_range > scan.range_max:
                continue

            theta = scan.angle_min + beam_idx * scan.angle_increment
            laser_x = beam_range * math.cos(theta)
            laser_y = beam_range * math.sin(theta)

            base_x = self.lidar_dx + laser_x * cos_yaw - laser_y * sin_yaw
            base_y = self.lidar_dy + laser_x * sin_yaw + laser_y * cos_yaw

            points.append((math.hypot(base_x, base_y), math.atan2(base_y, base_x)))

        return points


    def get_closest_obstacle_in_sector(self, points, center_bearing, half_width):
        ''' 
        Description:    Nearest obstacle distance in a cone ahead, within the forward cone.

        Args:
            points (list):              (distance, bearing) tuples, bearing in radians, 0 = straight ahead of the base, positive = to the left
            center_bearing (float):     Center of the cone, in radians, 0 = straight ahead of the base, positive = to the left
            half_width (float):         Half-width of the cone, in radians

        Returns:
            nearest (float):            Nearest obstacle distance in the cone, or math.inf if none
        '''
        nearest = math.inf # initialize nearest distance to infinity

        # checking each point in the points list to see if it is within the cone defined by center_bearing and half_width
        for dist, bearing in points:
            if abs(self.normalize_angle(bearing - center_bearing)) <= half_width:
                nearest = min(nearest, dist)
        return nearest


    def measure_side_clearances(self, points, half_width):
        '''
        Description:    Nearest obstacle distance on each side of straight-ahead, within
                        the forward cone. A side with nothing in it reads as open (inf).

        Args:
            points (list):      (distance, bearing) tuples, bearing in radians, 0 = straight ahead of the base, positive = to the left
            half_width (float): Half-width of the cone, in radians

        Returns:
            (left_room, right_room) (float, float)
        '''
        left_room = math.inf # initialize left room distance to infinity
        right_room = math.inf # initialize right room distance to infinity

        # checking each point in the points list to see if it is within the left or right half of the cone defined by half_width
        for dist, bearing in points:
            if 0.0 <= bearing <= half_width:
                left_room = min(left_room, dist)
            elif -half_width <= bearing < 0.0:
                right_room = min(right_room, dist)
        return left_room, right_room


    @staticmethod
    def normalize_angle(angle):
        '''
        Description:    Wrap an angle to [-pi, pi]

        Args:
            angle (float):          Angle in radians

        Returns:
            wrapped_angle (float):  Angle wrapped to [-pi, pi]
        '''
        return math.atan2(math.sin(angle), math.cos(angle))


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
        Description:    Timer function used to drive the eBot along the route.

        Args:
        Returns:
        '''

        # If any of the required data (route, odometry, or scan) is not available, publish zero velocities and return early
        if self.route is None or self.odom is None or self.scan is None:
            self.stop_ebot() # publish zero velocities to stop the robot
            return

        # Process navigation logic in a try-except block to catch any exceptions and log them
        try:
            # Ensure the lidar transform is available before proceeding with navigation
            if not self.lookup_laser_offset():
                self.stop_ebot()
                return

            # If all waypoints have been reached, log the completion message once and stop the robot
            if self.current_waypoint_index >= len(self.route):
                if not self.route_completed:
                    self.get_logger().info("All waypoints reached. Stopping.")
                    self.route_completed = True
                self.stop_ebot()
                return

            rover_x, rover_y, rover_yaw = self.odom # unpack the current odometry data (position and heading)
            target_x, target_y = self.route[self.current_waypoint_index] # get the target waypoint coordinates

            dx_to_goal = target_x - rover_x # calculate the difference in x between the target waypoint and the current position
            dy_to_goal = target_y - rover_y # calculate the difference in y between the target waypoint and the current position
            
            distance = math.hypot(dx_to_goal, dy_to_goal) # calculate the Euclidean distance to the target waypoint
            goal_bearing = self.normalize_angle(math.atan2(dy_to_goal, dx_to_goal) - rover_yaw) # calculate the bearing to the target waypoint relative to the current heading

            # If the distance to the target waypoint is within the arrival tolerance, increment the waypoint index and log the waypoint reached message
            if distance <= WAYPOINT_REACHED_DIST:
                self.current_waypoint_index += 1
                self.get_logger().info(
                    f"Waypoint {self.current_waypoint_index}/{len(self.route)} reached "
                    f"(distance {distance:.3f} m)")
                self.stop_ebot()
                return

            scan_points = self.get_base_frame_obstacles() # get the valid scan points in the base frame
            nearest_ahead = self.get_closest_obstacle_in_sector(scan_points, 0.0, FRONT_SECTOR_ANGLE) # find the nearest obstacle distance in the forward cone

            # If the nearest obstacle is closer than the safety distance, check the room on each side and decide how to steer around it
            if nearest_ahead < OBSTACLE_TRIGGER_DIST:
                left_room, right_room = self.measure_side_clearances(scan_points, FRONT_SECTOR_ANGLE)
                left_ok = left_room >= REQUIRED_SIDE_CLEARANCE # check if there is enough room on the left side
                right_ok = right_room >= REQUIRED_SIDE_CLEARANCE # check if there is enough room on the right side

                if not left_ok and not right_ok:
                    linear = 0.0 # stop moving forward if there is not enough room on either side
                    angular = STEER_AVOID_KP if left_room >= right_room else -STEER_AVOID_KP # turn away from the nearest obstacle based on which side has more room
                else:
                    if left_ok and right_ok and abs(left_room - right_room) < CLEARANCE_DIFF_TOLERANCE:
                        steer_bearing = FRONT_SECTOR_ANGLE if goal_bearing >= 0.0 else -FRONT_SECTOR_ANGLE # if both sides are okay and the room is similar, steer towards the goal bearing
                    elif left_ok and (not right_ok or left_room >= right_room):
                        steer_bearing = FRONT_SECTOR_ANGLE # steer towards the left if there is more room on the left side
                    else:
                        steer_bearing = -FRONT_SECTOR_ANGLE # steer towards the right if there is more room on the right side

                    heading_error = steer_bearing # calculate the heading error to the steering direction
                    linear = LINEAR_KP * nearest_ahead * max(0.0, math.cos(heading_error)) # calculate the linear velocity based on the nearest obstacle distance and the heading error
                    linear = min(linear, AVOID_SPEED_LIMIT) # limit the linear velocity to the maximum allowed when avoiding obstacles
                    angular = STEER_AVOID_KP * heading_error # calculate the angular velocity based on the heading error and the avoidance turn gain
            else:
                heading_error = goal_bearing # calculate the heading error to the target waypoint
                linear = LINEAR_KP * distance * max(0.0, math.cos(heading_error)) # calculate the linear velocity based on the distance to the target waypoint and the heading error
                angular = ANGULAR_KP * heading_error # calculate the angular velocity based on the heading error

            self.send_velocity(linear, angular) # publish the calculated linear and angular velocities to the cmd_vel topic to drive the robot towards the target waypoint while avoiding obstacles

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
