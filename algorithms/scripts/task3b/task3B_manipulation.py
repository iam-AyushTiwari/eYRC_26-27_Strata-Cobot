#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 3B of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:		    task3B_manipulation.py
# Functions:		main
# Nodes:		    Publishing Topics  - [ /delta_twist_cmds, /delta_joint_cmds, /package_picked ]
#                   Subscribing Topics - [ /tcp_pose_raw, /joint_states, /arm_status, /ur7e/end_effector/force_magnitude, /ebot_at_arm, /tf ]
#                   Service Clients    - [ /magnet, /ur_arm_controller/switch_controller, /ur_arm_controller/unlock_protective_stop ]


################### IMPORT MODULES #######################

import rclpy
import rclpy.executors
from collections import deque
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
import tf2_ros
from tf2_ros import TransformException
from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import Float32, Bool
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy


##################### TASK CONSTANTS #######################

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

# The handover with the eBot, on two held topics of our own (see task3B_planning.py)
at_arm_topic = '/ebot_at_arm'                # True while the eBot waits beside the arm with the package
package_picked_topic = '/package_picked'     # we publish True once the package is off the deck
package_frame = 'ore_package'                # the package's transform, from task3B_perception.py

# The ore crusher's hopper, measured from the depth image in base_link: far rim at
# x = -0.66, side walls at y = +/-0.27, rim at z = 0.21 and floor at z = 0.03.
hopper_drop = (-0.78, 0.0, 0.33)             # tool position the package is released at
hopper_hover_z = 0.42                        # tool height above the hopper before and after the drop

package_held_force = 2.0                     # N; the package reads about 2.94 N on the magnet, an ore about 1.46 N
package_clear_height = 0.08                  # m the tool rises above the grip point before the eBot may leave
package_max_linear = 0.10                    # m/s; slower moves while the heavier package hangs on the tool
force_filter_length = 5                      # readings in the running median of the noisy tool force

# Shared latched QoS for the handover topics
latched_qos = QoSProfile(
    depth=1,
    history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


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

        # task2a: transform listener for transforming from perception frame to base_link frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # task2a: service client for magnetic gripper
        self.magnet_cli = self.create_client(SetBool, '/magnet')
        self.get_logger().info('Waiting for /magnet service (non-blocking)...')

        self.tool_force = 0.0                                                           # median of the latest force readings (from force_callback())
        self.force_readings = deque(maxlen=force_filter_length)                         # latest raw force readings
        self.force_sub = self.create_subscription(Float32, '/ur7e/end_effector/force_magnitude', self.force_callback, 10)

        # the handover with the eBot
        self.at_arm_sub = self.create_subscription(Bool, at_arm_topic, self.atarmcb, latched_qos)
        self.picked_pub = self.create_publisher(Bool, package_picked_topic, latched_qos)     # tells the eBot it may drive home
            
        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05                                                             # rate of time to run one control cycle (seconds)
        self.switch_cli = self.create_client(                                           # client used to pick which command topic is live
            SwitchController, f'{servo_ns}/switch_controller')
        self.timer = self.create_timer(control_rate, self.process_waypoints)            # creating a timer based function which gets called on every 0.05 seconds (as defined by 'control_rate' variable)

        self.tcp_pose = None                                                            # tool pose variable (from tcpposecb())
        self.joint_angles = None                                                        # joint feedback variable (from jointstatecb())
        self.arm_status = None                                                          # arm state code variable (from armstatuscb())

        # Task 3B Ore Targets and Containers
        self.ore_targets = [
            'azurite_ore_1', 'azurite_ore_2',
            'malachite_ore_1', 'malachite_ore_2',
            'vanadinite_ore_1', 'vanadinite_ore_2'
        ]
        self.current_ore_idx = 0       # current ore index to pick

        self.ore_pickup_offset = 0.06 # how much below the ore to hover for pickup
        
        # Define container drop positions relative to base_link
        self.containers = {
            'azurite_ore': (-0.05, 0.58, 0.12),     # Blue container (Top)
            'malachite_ore': (-0.34, 0.58, 0.12),   # Green container (Middle)
            'vanadinite_ore': (-0.58, 0.58, 0.12)   # Orange container (Bottom / near cab)
        }

        # Transit waypoints routing around the front to avoid self-collision with the UR7e base
        self.gate_conveyor  = np.array([0.15, -0.45, 0.35])
        self.gate_front     = np.array([0.35, 0.0, 0.35])
        self.gate_container = np.array([0.15, 0.45, 0.35])
        self.gate_hopper    = np.array([-0.40, 0.42, 0.42])   # behind the containers' gate, on the way round to the hopper

        # The route the tool takes from a pick to its drop, and back the other way
        self.route = []                     # list of gate positions for the current transit
        self.route_idx = 0                  # index of the gate being driven to

        # The ore package
        self.package_waiting = False        # True while the eBot waits beside the arm (from atarmcb())
        self.package_done = False           # set once the package has been dropped in the hopper
        self.package_picked_sent = False    # set once /package_picked has been published True
        self.grip_z = None                  # tool height when the package was gripped
        self.publish_package_picked(False)

        self.task_state = 'APPROACH_HOVER'
        self.state_start_time = None

        # proportional controller gains 
        self.kp_linear = 1.2                # proportional gains for the linear velocity 
        self.kp_angular = 0.2               # proportional gains for the angular velocity
        self.distance_tolerance = 0.025     # metres; within this the arm is considered "at target"
        self.descend_tolerance = 0.004      # metres; tight capture gap (< 5.0 mm capture window)
        self.transit_tolerance = 0.06       # metres; fly-through gate tolerance
        self.contact_force = 2.5            # baseline gravity is ~1.46 N; contact threshold set to 2.5 N
        
        self.target_orientation = None      # orientation lock for the wrist
        self.unlock_cli = self.create_client(Trigger, f'{servo_ns}/unlock_protective_stop')

        # Controller switch management 
        self.active_controller = None       # active controller state variables
        self.switch_future = None           # switch future state
        self.switch_pending = None          # switch pending state 

        self.phase = 'unfold'               # phase state variables

        # remembered lift position, ore-missing timer and progress watchdog
        self.lift_xy = None                 # tool x,y at the moment the ore was gripped
        self.missing_since = None           # when the current ore's transform first went missing
        self.best_state = None              # watchdog: state being watched
        self.best_dist = None               # watchdog: smallest distance seen in that state
        self.best_time = None               # watchdog: when that smallest distance was seen

    def call_magnet_service(self, value: bool):
        '''
        Description:    Callback function for the magnet service.
                        Use this function to activate or deactivate the magnet.

        Args:
            value (bool):    True to activate the magnet, False to deactivate the magnet

        Returns:
        '''
        if not self.magnet_cli.service_is_ready():
            self.get_logger().warn('Magnet service not ready yet, skipping call', throttle_duration_sec=2.0)
            return
        request = SetBool.Request()
        request.data = value
        self.magnet_cli.call_async(request)

    def atarmcb(self, data):
        '''
        Description:    Callback function for /ebot_at_arm, published by the eBot planner.

        Args:
            data (Bool):    True while the eBot waits beside the arm with the package on its deck

        Returns:
        '''
        if data.data and not self.package_waiting:
            self.get_logger().info("eBot is beside the arm with the ore package")
        self.package_waiting = bool(data.data)

    def publish_package_picked(self, value):
        '''
        Description:    Publishes whether the package is off the eBot's deck.

        Args:
            value (bool):    True once the package is lifted clear of the deck

        Returns:
        '''
        msg = Bool()
        msg.data = bool(value)
        self.picked_pub.publish(msg)

    def queue_package(self):
        '''
        Description:    Puts the package next in the job list once the eBot is waiting with it,
                        but never in the middle of moving an ore.

        Args:
        Returns:
        '''
        if not self.package_waiting or self.package_done or package_frame in self.ore_targets:
            return
        if self.task_state != 'APPROACH_HOVER':
            return
        self.ore_targets.insert(self.current_ore_idx, package_frame)
        self.missing_since = None
        self.get_logger().info("Package queued as the next job")

    def force_callback(self, msg):
        '''
        Description:    Callback function for the force sensor topic.
                        Use this function to receive the force on the tool.

        Args:
            msg (Float32):    Force on the tool

        Returns:
        '''
        # a running median, so one noisy reading can neither fake a contact nor a dropped package
        self.force_readings.append(msg.data)
        self.tool_force = float(np.median(self.force_readings))

    def tcpposecb(self, data):
        '''
        Description:    Callback function for the tool pose topic.
                        Use this function to receive where the tool currently is.

        Args:
            data (PoseStamped):    Pose of the tool, reported in base_link

        Returns:
        '''
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
        if 20 <= data.data <= 23 and self.arm_status != data.data:
            self.get_logger().error(
                f"PROTECTIVE STOP: arm_status={data.data} in state {self.task_state}, "
                f"force={self.tool_force:.2f} N")
            if self.unlock_cli.service_is_ready():
                req = Trigger.Request()
                self.unlock_cli.call_async(req)
                self.get_logger().info("Requested /ur_arm_controller/unlock_protective_stop")
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
        req = SwitchController.Request()
        req.activate_controllers = [controller]
        req.deactivate_controllers = [joint_controller if controller == twist_controller else twist_controller]
        req.strictness = SwitchController.Request.STRICT

        if not self.switch_cli.service_is_ready():
            self.get_logger().error("switch controller is not ready")
            return False

        self.switch_future = self.switch_cli.call_async(req)
        self.switch_pending = controller
        return True

    def publish_zero_twist(self):
        '''
        Description:    Publish zero twist commands to the arm to prevent coasting and satisfy the command timeout.
        Args:
        Returns:
        '''
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

    def publish_zero_joint(self):
        '''
        Description:    Publish zero joint velocities to the arm to prevent coasting and satisfy the command timeout.
        Args:
        Returns:
        '''
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        msg.velocities = [0.0] * len(joint_names)
        self.joint_pub.publish(msg)

    def run_unfold(self):
        '''
        Description:    Drive the joints toward unfold_targets using switch_controller.
        Args:
        Returns:
        '''
        if self.active_controller != joint_controller:
            if self.switch_future is None:
                self.switch_controller(joint_controller)
            elif self.switch_future.done():
                try:
                    res = self.switch_future.result()
                    if res.ok:
                        self.active_controller = self.switch_pending
                except Exception:
                    pass
                self.switch_future = None
            self.publish_zero_twist()
            return

        errors = [unfold_targets[name] - angle
                  for name, angle in zip(joint_names, self.joint_angles)]

        if max(abs(e) for e in errors) < unfold_tolerance_rad:
            self.publish_zero_joint()
            self.target_orientation = self.tcp_pose.orientation 
            self.phase = 'track'
            self.get_logger().info("Unfold complete, switching to waypoint tracking")
            return

        cap = cap_joint_rps * 0.98
        velocities = [max(-cap, min(cap, kp_joint * e)) for e in errors]

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        msg.velocities = velocities
        self.joint_pub.publish(msg)

    def drop_position(self, job_name):
        '''
        Description:    Where the tool releases the item of a job: the container for an ore's
                        kind, or the crusher's hopper for the package.

        Args:
            job_name (str):     transform name of the item, e.g. 'azurite_ore_1' or 'ore_package'

        Returns:
            drop (ndarray):     tool position (x, y, z) at release, in base_link
            hover_z (float):    tool height above the drop before lowering and after release
        '''
        if job_name == package_frame:
            return np.array(hopper_drop), hopper_hover_z
        kind = job_name.rsplit('_', 1)[0]
        return np.array(self.containers.get(kind, (-0.34, 0.58, 0.12))), 0.35

    def route_to_drop(self, job_name):
        '''
        Description:    The gates the tool passes through from a lifted item to its drop.

        Args:
            job_name (str):     transform name of the item being carried

        Returns:
            route (list):       gate positions (ndarray), in order
        '''
        if job_name == package_frame:
            return [self.gate_front, self.gate_container, self.gate_hopper]
        return [self.gate_conveyor, self.gate_front, self.gate_container]

    def run_tracking(self):
        '''
        Description:    Pick-and-place state machine, using TF lookups, for the ores and the
                        ore package. Each job runs APPROACH_HOVER, DESCEND_ORE, GRIP_ORE,
                        LIFT_ORE, TRANSIT, HOVER_CONTAINER, LOWER_CONTAINER, RELEASE_ORE,
                        LIFT_FROM_CONTAINER and RETURN.
        Args:
        Returns:
        '''
        if self.active_controller != twist_controller:
            if self.switch_future is None:
                self.switch_controller(twist_controller)
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

        # the package goes next as soon as the eBot is waiting with it
        self.queue_package()

        if self.current_ore_idx >= len(self.ore_targets):
            self.publish_zero_twist()
            if self.package_done:
                self.get_logger().info("All ores and the package processed successfully!", throttle_duration_sec=5.0)
            else:
                self.get_logger().info("All ores processed; waiting for the ore package", throttle_duration_sec=5.0)
            return

        current_ore_name = self.ore_targets[self.current_ore_idx]
        is_package = current_ore_name == package_frame

        now = self.get_clock().now()

        if self.task_state in ('APPROACH_HOVER', 'DESCEND_ORE'):
            try:
                trans = self.tf_buffer.lookup_transform(
                    base_frame, current_ore_name, rclpy.time.Time()
                )
                ore_x = trans.transform.translation.x
                ore_y = trans.transform.translation.y
                ore_z = trans.transform.translation.z
                self.missing_since = None
            except TransformException as ex:
                self.get_logger().warn(f"Waiting for transform {current_ore_name}: {ex}", throttle_duration_sec=2.0)
                self.publish_zero_twist()
                # the package is waited for however long it takes; an absent ore is skipped
                if is_package:
                    return
                if self.missing_since is None:
                    self.missing_since = now
                if (now - self.missing_since).nanoseconds / 1e9 > 3.0:
                    found_next = False
                    for future_ore in self.ore_targets[self.current_ore_idx + 1:]:
                        if self.tf_buffer.can_transform(base_frame, future_ore, rclpy.time.Time()):
                            self.get_logger().info(f"Skipping absent ore {current_ore_name} -> {future_ore}")
                            self.current_ore_idx = self.ore_targets.index(future_ore)
                            self.task_state = 'APPROACH_HOVER'
                            self.missing_since = None
                            found_next = True
                            break
                    if not found_next and (now - self.missing_since).nanoseconds / 1e9 > 5.0:
                        self.get_logger().info(f"No future ores found on TF, advancing past {current_ore_name}")
                        self.current_ore_idx += 1
                        self.missing_since = None
                return

        drop, drop_hover_z = self.drop_position(current_ore_name)

        # State 1: Hover above the item
        if self.task_state == 'APPROACH_HOVER':
            target_pos = np.array([ore_x, ore_y, 0.35])

        # State 2: Descend onto the item (target lower so the magnet makes solid contact with its top surface)
        elif self.task_state == 'DESCEND_ORE':
            target_pos = np.array([ore_x, ore_y, ore_z + 0.035])
            self.call_magnet_service(True)

        # State 3: Dwell to engage magnet
        elif self.task_state == 'GRIP_ORE':
            self.publish_zero_twist()
            self.call_magnet_service(True)
            if self.state_start_time is None:
                self.state_start_time = now
            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 1.2:
                self.state_start_time = None
                self.best_dist = None
                self.best_time = None
                self.lift_xy = (self.tcp_pose.position.x, self.tcp_pose.position.y)
                self.grip_z = self.tcp_pose.position.z
                self.task_state = 'LIFT_ORE'
                self.get_logger().info(
                    f"Magnet engaged for {current_ore_name}, force={self.tool_force:.2f} N, lifting...")
            return

        # State 4: Lift vertically
        elif self.task_state == 'LIFT_ORE':
            target_pos = np.array([self.lift_xy[0], self.lift_xy[1], 0.35])
            # the eBot may leave as soon as the package hangs clear of its deck
            if (is_package and not self.package_picked_sent
                    and self.tcp_pose.position.z > self.grip_z + package_clear_height
                    and self.tool_force > package_held_force):
                self.publish_package_picked(True)
                self.package_picked_sent = True
                self.get_logger().info(f"Package clear of the deck (force={self.tool_force:.2f} N); eBot may leave")

        # State 5: Through the gates to the drop
        elif self.task_state in ('TRANSIT', 'RETURN'):
            target_pos = self.route[self.route_idx]

        # State 6: Hover above the container or hopper
        elif self.task_state == 'HOVER_CONTAINER':
            target_pos = np.array([drop[0], drop[1], drop_hover_z])

        # State 7: Lower into the container or hopper
        elif self.task_state == 'LOWER_CONTAINER':
            target_pos = drop

        # State 8: release the item
        elif self.task_state == 'RELEASE_ORE':
            self.publish_zero_twist()
            self.call_magnet_service(False)
            if self.state_start_time is None:
                self.state_start_time = now
            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 0.6:
                self.state_start_time = None
                self.best_dist = None
                self.best_time = None
                self.get_logger().info(f"Dropped {current_ore_name}.")
                if is_package:
                    self.package_done = True
                self.task_state = 'LIFT_FROM_CONTAINER'
            return

        # State 9: Lift vertically out of the container or hopper
        elif self.task_state == 'LIFT_FROM_CONTAINER':
            target_pos = np.array([drop[0], drop[1], drop_hover_z])

        current_pos = np.array([self.tcp_pose.position.x, self.tcp_pose.position.y, self.tcp_pose.position.z])
        error_pos = target_pos - current_pos
        distance = np.linalg.norm(error_pos)

        self.get_logger().info(
            f"State: {self.task_state} | Item: {current_ore_name} | Distance: {distance:.4f} m",
            throttle_duration_sec=0.5
        )

        if self.best_state != self.task_state or self.best_dist is None or distance < self.best_dist - 0.0002:
            self.best_state, self.best_dist, self.best_time = self.task_state, distance, now
        stalled = (now - self.best_time).nanoseconds / 1e9 > 3.0
        if stalled:
            self.get_logger().warn(
                f"STALLED in {self.task_state} | dist={distance:.4f} m | "
                f"arm_status={self.arm_status} | force={self.tool_force:.2f} N",
                throttle_duration_sec=2.0)

        if self.task_state == 'DESCEND_ORE':
            tol = self.descend_tolerance
        elif self.task_state in ('TRANSIT', 'RETURN'):
            tol = self.transit_tolerance
        else:
            tol = self.distance_tolerance

        force_contact = (self.task_state == 'DESCEND_ORE' and
                         (self.arm_status == 12 or self.tool_force > self.contact_force or (stalled and distance < 0.012)))

        if distance > tol and not force_contact:
            # slow near contact, and while the heavier package hangs on the tool
            if self.task_state == 'DESCEND_ORE':
                max_linear = 0.030
            elif is_package and self.task_state in ('LIFT_ORE', 'TRANSIT', 'HOVER_CONTAINER', 'LOWER_CONTAINER'):
                max_linear = package_max_linear
            else:
                max_linear = cap_linear_mps
            v_lin = self.kp_linear * error_pos
            v_mag = np.linalg.norm(v_lin)
            max_cap = max_linear * 0.98

            if v_mag > max_cap:
                v_lin = v_lin * (max_cap / v_mag)

            q_curr = [self.tcp_pose.orientation.x, self.tcp_pose.orientation.y, self.tcp_pose.orientation.z, self.tcp_pose.orientation.w]
            R_curr = Rotation.from_quat(q_curr).as_matrix()
            a_curr = R_curr[:, 2]

            q_target = [self.target_orientation.x, self.target_orientation.y, self.target_orientation.z, self.target_orientation.w]
            R_target = Rotation.from_quat(q_target).as_matrix()
            b_target = R_target[:, 2]

            cross_product = np.cross(a_curr, b_target)
            c_norm = np.linalg.norm(cross_product)
            dot_a_b = np.dot(a_curr, b_target)

            if c_norm > 1e-6:
                error = (cross_product / c_norm) * math.atan2(c_norm, dot_a_b)
                omega = self.kp_angular * error
            else:
                omega = np.zeros(3)

            omega_mag = np.linalg.norm(omega)
            max_ang_cap = cap_angular_rps * 0.98
            if omega_mag > max_ang_cap:
                omega = omega * (max_ang_cap / omega_mag)

            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = base_frame

            twist_msg.twist.linear.x = float(v_lin[0])
            twist_msg.twist.linear.y = float(v_lin[1])
            twist_msg.twist.linear.z = float(v_lin[2])

            twist_msg.twist.angular.x = float(omega[0])
            twist_msg.twist.angular.y = float(omega[1])
            twist_msg.twist.angular.z = float(omega[2])

            self.twist_pub.publish(twist_msg)
        else:
            self.publish_zero_twist()
            self.best_dist = None
            self.best_time = None
            if self.task_state == 'APPROACH_HOVER':
                self.get_logger().info(f"Hovering above {current_ore_name}, descending to grasp...")
                self.task_state = 'DESCEND_ORE'
            elif self.task_state == 'DESCEND_ORE':
                self.get_logger().info(f"Reached {current_ore_name}, activating magnet...")
                self.call_magnet_service(True)
                self.state_start_time = None
                self.task_state = 'GRIP_ORE'
            elif self.task_state == 'LIFT_ORE':
                # a package that did not come up with the tool is gripped again
                if is_package and self.tool_force < package_held_force:
                    self.get_logger().warn(
                        f"Package not held (force={self.tool_force:.2f} N); trying the grip again")
                    self.call_magnet_service(False)
                    self.task_state = 'APPROACH_HOVER'
                    return
                self.get_logger().info(f"Lifted {current_ore_name}, transiting to its drop...")
                self.route = self.route_to_drop(current_ore_name)
                self.route_idx = 0
                self.task_state = 'TRANSIT'
            elif self.task_state == 'TRANSIT':
                self.route_idx += 1
                if self.route_idx >= len(self.route):
                    self.task_state = 'HOVER_CONTAINER'
            elif self.task_state == 'HOVER_CONTAINER':
                self.get_logger().info(f"Hovering over the drop for {current_ore_name}, lowering...")
                self.task_state = 'LOWER_CONTAINER'
            elif self.task_state == 'LOWER_CONTAINER':
                self.get_logger().info(f"Reached the drop for {current_ore_name}, releasing...")
                self.call_magnet_service(False)
                self.state_start_time = None
                self.task_state = 'RELEASE_ORE'
            elif self.task_state == 'LIFT_FROM_CONTAINER':
                self.get_logger().info("Lifted clear of the drop, returning...")
                self.route = list(reversed(self.route_to_drop(current_ore_name)))
                self.route_idx = 0
                self.task_state = 'RETURN'
            elif self.task_state == 'RETURN':
                self.route_idx += 1
                if self.route_idx >= len(self.route):
                    self.current_ore_idx += 1
                    self.task_state = 'APPROACH_HOVER'

    def process_waypoints(self):
        '''
        Description:    Timer function used to drive the tool through the waypoints.
        Args:
        Returns:
        '''
        if self.tcp_pose is None or self.joint_angles is None or self.arm_status is None:
            self.publish_zero_twist()
            return

        if 20 <= self.arm_status <= 23:
            self.publish_zero_twist()
            self.get_logger().warn(
                f"Arm latched (arm_status={self.arm_status}); commands are ignored until it is unlocked",
                throttle_duration_sec=2.0)
            return
        
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
    rclpy.init(args=sys.argv)
    node = arm_waypoints()                              # creating a new object for class 'arm_waypoints'
    try:
        rclpy.spin(node)                                # spinning on the object to make it alive in ROS 2 DDS
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass                                            # Ctrl + C between the two runs ends the node quietly
    finally:
        node.destroy_node()                             # destroy node after spin ends
        if rclpy.ok():
            rclpy.shutdown()                            # shutdown process


if __name__ == '__main__':
    main()