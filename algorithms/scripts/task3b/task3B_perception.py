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
# Author List:      Aryan Joshi
# Filename:         task3B_perception.py
# Functions:        detect_ores, rotation_from_quat, main
# Nodes:            Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/camera/color/image_raw, /camera/camera/aligned_depth_to_color/image_raw, /camera/camera/color/camera_info, /ebot_at_arm, /package_picked, /tf ]


################### IMPORT MODULES #######################

import rclpy
import rclpy.executors
import sys
import cv2
import math
import tf2_ros
import numpy as np
from collections import deque
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image
import tf2_geometry_msgs
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy


# Two ores of each type are spawned - six in all - told apart by an id of 1 or 2.
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

# The RealSense topics. The depth image is ALIGNED to the colour image.
color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'

# The parent frame every ore transform is published against.
base_frame = 'base_link'

# HSV ranges for different ore types
hsv_ranges = {
    'azurite_ore':    ((95, 140, 60),  (118, 255, 255)),
    'malachite_ore':  ((60, 140, 60),  (85, 255, 255)),
    # vanadinite is fully saturated and bright (about H 10, S 255, V 255); the rocks and the
    # terrain share its hue but sit at S 155-215 and V 50-175, so they are cut by S and V
    'vanadinite_ore': ((0, 235, 180),  (15, 255, 255)),
}
vanadinite_wrap_range = ((170, 235, 180), (179, 255, 255))

# Each ore's transform sits on the centre of its top face, the point detection is scored
# against (within 0.05 m); at the cube's centre it would start 0.038 m off.
EXPECTED_ORES = 4

MIN_ORE_AREA = 60           # px; an ore at the conveyor's far end reads about 140 px, and the conveyor zone filters the noise
MAX_ORE_AREA = 2500         # tightened to filter terrain
# The conveyor, in base_link, where every ore starts (ore tops measured at z = -0.064,
# y = -0.62 to -0.81, x = -0.25 to 0.05). A detection whose top is anywhere else, on a rock,
# the terrain or an ore already dropped in a container, is ignored.
ORE_ZONE_X = (-0.70, 0.70)
ORE_ZONE_Y = (-1.05, -0.30)
ORE_ZONE_Z = (-0.15, 0.05)

MATCH_PX = 45               # stricter matching threshold to prevent ID shifting
SAMPLES_PER_ORE = 15        # more samples, so the median rides out the noisier depth of Task 3B
DEPTH_HALF_WINDOW = 4
DEPTH_MEDIAN_KERNEL = 5     # px, median filter over the depth image before it is read

FREEZE_WHEN_STABLE = True
FREEZE_SPREAD = 0.010
FREEZE_AFTER_CYCLES = 25    # cycles with every buffer full after which the medians are frozen anyway
SHOW_WINDOW = True     
SAVE_CALIBRATION_FRAME = False

# The ore package, detected on the eBot's deck from the depth image once the eBot is
# waiting beside the arm. Its transform sits on the centre of its top face, the point its
# detection is scored against.
package_frame = 'ore_package'
ebot_frame = 'ebot_base_link'                 # the eBot's base, in the same TF tree as base_link
at_arm_topic = '/ebot_at_arm'                 # True while the eBot waits beside the arm (task3B_planning.py)
package_picked_topic = '/package_picked'      # True once the arm has the package (task3B_manipulation.py)
PACKAGE_SEARCH_RADIUS = 0.40                  # m, around the eBot's centre, in the ground plane
# The eBot's top deck is the top of its tower, 0.675 m above ebot_base_link (collision box
# 0.35 x 0.29 x 0.435 m centred 0.4575 m up), so the box's top face is about 0.751 m up.
PACKAGE_MIN_ABOVE_BASE = 0.70                 # m above ebot_base_link: below this is the tower's own top
PACKAGE_MAX_ABOVE_BASE = 0.82                 # m above ebot_base_link: above this is the arm, not the box
PACKAGE_TOP_RANK = 100                        # the top face's height is the 100th-highest point, so up to 99 noisy readings above it are ignored
PACKAGE_TOP_BELOW = 0.006                     # m below the top face's height a point may sit and still count as on it;
                                              # more takes in the near side face and drags the centre 1-3 cm towards the camera
PACKAGE_TOP_ABOVE = 0.010                     # m above it, for the depth noise
PACKAGE_MIN_POINTS = 150                      # fewer top-face points than this is not the box
PACKAGE_SAMPLES = 10                          # stable samples before the package position is frozen
PACKAGE_FREEZE_SPREAD = 0.020                 # m, largest spread among those samples
PACKAGE_MAX_ATTEMPTS = 25                     # after this many samples the median of the latest is frozen anyway

latched_qos = QoSProfile(
    depth=1,
    history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


##################### FUNCTION DEFINITIONS #######################

def detect_ores(image):
    '''
    Description:    Function to detect the ores present in a colour image frame and
                    return the pixel location and the type of each one found.

    Args:
        image                   (Image):    Input colour image frame received from the camera topic

    Returns:
        center_ore_list         (list):     Center pixel (cX, cY) of every ore detected in the frame
        ore_type_list           (list):     Type of each ore detected, taken from 'ore_types'
    '''
    center_ore_list = []
    ore_type_list = []

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for ore_type, (lower, upper) in hsv_ranges.items():
        mask = cv2.inRange(hsv_image,
                           np.array(lower, dtype=np.uint8),
                           np.array(upper, dtype=np.uint8))

        if ore_type == 'vanadinite_ore':
            mask_wrap = cv2.inRange(hsv_image,
                                    np.array(vanadinite_wrap_range[0], dtype=np.uint8),
                                    np.array(vanadinite_wrap_range[1], dtype=np.uint8))
            mask = cv2.bitwise_or(mask, mask_wrap)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            a = cv2.contourArea(c)
            if not (MIN_ORE_AREA <= a <= MAX_ORE_AREA):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if 0.5 <= float(w) / h <= 2.0:
                candidates.append(c)

        candidates.sort(key=cv2.contourArea, reverse=True)

        # every candidate is returned, largest first: the node keeps those on the conveyor, two of each kind at most
        for cnt in candidates:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            center_ore_list.append((cX, cY))
            ore_type_list.append(ore_type)

    return center_ore_list, ore_type_list


def rotation_from_quat(qx, qy, qz, qw):
    '''
    Description:    Rotation matrix of a unit quaternion.

    Args:
        qx, qy, qz, qw (float):    the quaternion

    Returns:
        rotation (ndarray):        3 x 3 rotation matrix
    '''
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


##################### CLASS DEFINITION #######################

class ore_tf(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to detect the ores in the cell and broadcast a transform for each one.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ore_tf
        '''
        super().__init__('ore_tf_publisher')

        self.color_cam_sub = self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.caminfocb, 10)
        self.at_arm_sub = self.create_subscription(Bool, at_arm_topic, self.atarmcb, latched_qos)
        self.picked_sub = self.create_subscription(Bool, package_picked_topic, self.packagepickedcb, latched_qos)
        
        image_processing_rate = 0.2
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(image_processing_rate, self.process_image)

        self.cv_image = None
        self.depth_image = None
        self.cam_info = None

        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        self.ore_registry = {ot: [] for ot in ore_types}
        self.ore_samples = {}
        self.frozen = False
        self.calibration_saved = False

        self.ebot_at_arm = False                                    # eBot waiting beside the arm (from atarmcb())
        self.package_picked = False                                 # arm has the package (from packagepickedcb())
        self.package_samples = deque(maxlen=PACKAGE_SAMPLES)        # recent (x, y, z) of the package in base_link
        self.package_pose = None                                    # frozen (x, y, z) of the package in base_link
        self.package_attempts = 0                                   # package samples taken so far
        self.full_cycles = 0                                        # cycles every ore buffer has been full

    def atarmcb(self, data):
        '''
        Description:    Callback function for /ebot_at_arm, published by the eBot planner.

        Args:
            data (Bool):    True while the eBot waits beside the arm with the package

        Returns:
        '''
        self.ebot_at_arm = bool(data.data)

    def packagepickedcb(self, data):
        '''
        Description:    Callback function for /package_picked, published by the arm node.

        Args:
            data (Bool):    True once the arm has lifted the package off the deck

        Returns:
        '''
        if data.data and not self.package_picked:
            self.get_logger().info("Package picked; no longer publishing its transform")
        self.package_picked = bool(data.data)

    def depthimagecb(self, data):
        '''
        Description:    Callback function for the aligned depth camera topic.
        Args:
            data (Image):    Input depth image frame received from the aligned depth camera topic
        Returns: 
        '''
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            if depth_cv.dtype == np.uint16:
                depth_m = depth_cv.astype(np.float32) / 1000.0
            else:
                depth_m = depth_cv.astype(np.float32)
            # no reading (nan or inf) becomes 0, which every reader skips; then a median
            # filter takes out the per-pixel noise of Task 3B's depth
            depth_m[~np.isfinite(depth_m)] = 0.0
            self.depth_image = cv2.medianBlur(depth_m, DEPTH_MEDIAN_KERNEL).astype(np.float64)
        except CvBridgeError as e:
            self.get_logger().error(f"Depth Image CvBridgeError: {e}")

    def colorimagecb(self, data):
        '''
        Description:    Callback function for the colour camera raw topic.
        Args:
            data (Image):    Input coloured raw image frame received from the image_raw camera topic
        Returns: 
        '''
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Color Image CvBridgeError: {e}")

    def caminfocb(self, data):
        '''
        Description:    Callback function for the camera info topic.
        Args:
            data (CameraInfo):    Camera calibration published by the camera
        Returns: 
        '''
        self.cam_info = data
        self.fx = data.k[0]
        self.cx = data.k[2]
        self.fy = data.k[4]
        self.cy = data.k[5]

    def _assign_ore_id(self, ore_type, cX, cY, used):
        '''
        Description:    Match a detection to an already-named ore of the same type by
                        pixel distance, or name it if it is new.

        Args:
            ore_type (str):   one of 'ore_types'
            cX, cY   (int):   centre pixel of the detection
            used     (set):   (ore_type, id) pairs already claimed this cycle

        Returns:
            1 or 2, or None if the detection cannot be one of the two ores of this type
        '''
        known = self.ore_registry[ore_type]
        free = [i for i in range(len(known)) if (ore_type, i + 1) not in used]

        if free:
            best = min(free, key=lambda i: math.hypot(cX - known[i][0], cY - known[i][1]))
            if math.hypot(cX - known[best][0], cY - known[best][1]) <= MATCH_PX:
                known[best] = (cX, cY)
                return best + 1

        if len(known) < 2:  # Up to 2 ores per type (_1 and _2)
            known.append((cX, cY))
            return len(known)

        return None

    def _ore_top_in_base(self, cX, cY, cam_to_base, optical_frame):
        '''
        Description:    Read depth at a detection's centre, deproject it and move it into base_frame.

        Args:
            cX, cY (int):                     centre pixel of the detection
            cam_to_base (TransformStamped):   transform from camera to base frame
            optical_frame (str):              frame id of the optical frame

        Returns:
            top (Point):    the detection's top surface in base_frame, or None without a depth reading
        '''
        rows, cols = self.depth_image.shape[:2]
        patch = self.depth_image[max(0, cY - DEPTH_HALF_WINDOW):min(rows, cY + DEPTH_HALF_WINDOW + 1),
                                 max(0, cX - DEPTH_HALF_WINDOW):min(cols, cX + DEPTH_HALF_WINDOW + 1)]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]

        if valid.size == 0:
            return None

        z = float(np.median(valid))
        point_in_camera = PointStamped()
        point_in_camera.header.frame_id = optical_frame
        point_in_camera.header.stamp = self.get_clock().now().to_msg()
        point_in_camera.point.x = float((cX - self.cx) * z / self.fx)
        point_in_camera.point.y = float((cY - self.cy) * z / self.fy)
        point_in_camera.point.z = z

        return tf2_geometry_msgs.do_transform_point(point_in_camera, cam_to_base).point

    def _in_ore_zone(self, top):
        '''
        Description:    Whether a detection's top surface lies on the conveyor, where the ores start.

        Args:
            top (Point):    the detection's top surface in base_frame

        Returns:
            inside (bool)
        '''
        return (ORE_ZONE_X[0] <= top.x <= ORE_ZONE_X[1]
                and ORE_ZONE_Y[0] <= top.y <= ORE_ZONE_Y[1]
                and ORE_ZONE_Z[0] <= top.z <= ORE_ZONE_Z[1])

    def _update_ore_position(self, name, top):
        '''
        Description:    Add a detection's position to the ore's recent samples.

        Args:
            name (str):     name of the ore, e.g. 'azurite_ore_1'
            top (Point):    the ore's top surface in base_frame, from _ore_top_in_base()

        Returns:
        '''
        buf = self.ore_samples.setdefault(name, deque(maxlen=SAMPLES_PER_ORE))
        buf.append((top.x, top.y, top.z))

    def _broadcast_ores(self):
        '''
        Description:    Broadcast every ore measured so far, every cycle, from the median
                        of its recent samples.

        Args:
        Returns:
        '''
        stamp = self.get_clock().now().to_msg()
        for name, buf in self.ore_samples.items():
            if len(buf) == 0:
                continue
            x, y, z = np.median(np.array(buf), axis=0)

            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = base_frame
            t.child_frame_id = name
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.translation.z = float(z)
            t.transform.rotation.w = 1.0
            self.br.sendTransform(t)

    def _ready_to_freeze(self):
        '''
        Description:    Whether every expected ore has a full buffer whose samples agree.

        Args:
        Returns:
            ready (bool)
        '''
        if not FREEZE_WHEN_STABLE:
            return False
 
        if len(self.ore_samples) != EXPECTED_ORES:
            return False
 
        for buf in self.ore_samples.values():
            if len(buf) < SAMPLES_PER_ORE:
                self.full_cycles = 0
                return False
        self.full_cycles += 1

        # with noisy depth the samples may never agree that closely; the medians are good anyway
        if self.full_cycles >= FREEZE_AFTER_CYCLES:
            return True
        for buf in self.ore_samples.values():
            if np.ptp(np.array(buf), axis=0).max() > FREEZE_SPREAD:
                return False
        return True
 
    def detect_package(self, cam_to_base):
        '''
        Description:    Finds the ore package on the eBot's deck in the depth image: the
                        highest flat surface above the eBot, below the arm. Adds its centre
                        to the package's recent samples, and freezes it once they agree.

        Args:
            cam_to_base (TransformStamped):   transform from the camera optical frame to base_link

        Returns:
            top_pixels (ndarray):   (row, col) of the top-face pixels found, for the debug window
        '''
        try:
            ebot = self.tf_buffer.lookup_transform(base_frame, ebot_frame, rclpy.time.Time()).transform.translation
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"eBot transform not available: {e}", throttle_duration_sec=2.0)
            return None

        # every pixel of the depth image as a point in base_link
        depth = self.depth_image
        rows, cols = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
        valid = np.isfinite(depth) & (depth > 0.0)
        z_cam = np.where(valid, depth, 0.0)
        cam_points = np.stack([(cols - self.cx) * z_cam / self.fx,
                               (rows - self.cy) * z_cam / self.fy,
                               z_cam], axis=-1)
        rot = cam_to_base.transform.rotation
        trans = cam_to_base.transform.translation
        cam_rotation = rotation_from_quat(rot.x, rot.y, rot.z, rot.w)
        base_points = cam_points @ cam_rotation.T + np.array([trans.x, trans.y, trans.z])

        # keep the points over the eBot, above its body and below the arm
        height = base_points[..., 2] - ebot.z
        near = (valid
                & (np.hypot(base_points[..., 0] - ebot.x, base_points[..., 1] - ebot.y) < PACKAGE_SEARCH_RADIUS)
                & (height > PACKAGE_MIN_ABOVE_BASE) & (height < PACKAGE_MAX_ABOVE_BASE))
        if np.count_nonzero(near) < PACKAGE_MIN_POINTS:
            return None

        # the top face is the highest surface left; the box covers only a small share of the
        # points, so its height is taken by rank, not by percentile
        heights = base_points[..., 2][near]
        rank = min(PACKAGE_TOP_RANK, heights.size)
        top_z = np.partition(heights, -rank)[-rank]
        top = near & (base_points[..., 2] > top_z - PACKAGE_TOP_BELOW) & (base_points[..., 2] < top_z + PACKAGE_TOP_ABOVE)
        if np.count_nonzero(top) < PACKAGE_MIN_POINTS:
            return None

        centre = np.median(base_points[top], axis=0)
        # the transform sits on the centre of the top face, the point detection is scored against
        self.package_samples.append((centre[0], centre[1], centre[2]))
        self.package_attempts += 1

        samples = np.array(self.package_samples)
        settled = np.ptp(samples, axis=0).max() <= PACKAGE_FREEZE_SPREAD
        if len(samples) == PACKAGE_SAMPLES and (settled or self.package_attempts >= PACKAGE_MAX_ATTEMPTS):
            self.package_pose = tuple(np.median(samples, axis=0))
            self.get_logger().info(
                f"Package found at ({self.package_pose[0]:.3f}, {self.package_pose[1]:.3f}, "
                f"{self.package_pose[2]:.3f}) in {base_frame}; frozen")
        return np.argwhere(top)

    def broadcast_package(self):
        '''
        Description:    Broadcasts the package's transform while it sits on the eBot beside the arm.

        Args:
        Returns:
        '''
        if self.package_pose is None or self.package_picked:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = base_frame
        t.child_frame_id = package_frame
        t.transform.translation.x = float(self.package_pose[0])
        t.transform.translation.y = float(self.package_pose[1])
        t.transform.translation.z = float(self.package_pose[2])
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)

    def process_image(self):
        '''
        Description:    Timer function used to detect the ores and the package and publish a
                        transform for each one.

        Args:
        Returns:
        '''
        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return

        optical_frame = self.cam_info.header.frame_id
 
        try:
            cam_to_base = self.tf_buffer.lookup_transform(
                base_frame, optical_frame, rclpy.time.Time())
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)
            cam_to_base = None
 
        vis_image = self.cv_image.copy()

        if not self.frozen and self._ready_to_freeze():
            self.frozen = True
            self.get_logger().info("All 4 ores stable! Freezing positions for gripping.")

        if not self.frozen and cam_to_base is not None:
            center_ore_list, ore_type_list = detect_ores(self.cv_image)
            used = set()
 
            for (cX, cY), ore_type in zip(center_ore_list, ore_type_list):
                # only what lies on the conveyor is an ore still to be sorted
                top = self._ore_top_in_base(cX, cY, cam_to_base, optical_frame)
                if top is None or not self._in_ore_zone(top):
                    continue
                ore_id = self._assign_ore_id(ore_type, cX, cY, used)
                if ore_id is None:
                    continue
                used.add((ore_type, ore_id))
                name = f"{ore_type}_{ore_id}"
 
                self._update_ore_position(name, top)
 
                box_size = 15
                x_axis, y_axis, width, height = cX - box_size, cY - box_size, box_size * 2, box_size * 2
                cv2.rectangle(vis_image, (x_axis, y_axis), (x_axis + width, y_axis + height), (0, 255, 0), 2)
                cv2.putText(vis_image, name, (x_axis, max(y_axis - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
 
        self._broadcast_ores()

        # the package, once the eBot is waiting beside the arm with it
        if (self.ebot_at_arm and not self.package_picked and self.package_pose is None
                and cam_to_base is not None):
            top_pixels = self.detect_package(cam_to_base)
            if top_pixels is not None:
                vis_image[top_pixels[:, 0], top_pixels[:, 1]] = (255, 0, 255)
        self.broadcast_package()
 
        if SHOW_WINDOW:
            cv2.imshow("Ore Detector", vis_image)
            cv2.waitKey(1)


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the ore_tf class to perform its task
    '''
    rclpy.init(args=sys.argv)
    node = ore_tf()                                     # creating a new object for class 'ore_tf'
    try:
        rclpy.spin(node)                                # spinning on the object to make it alive in ROS 2 DDS
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass                                            # Ctrl + C between the two runs ends the node quietly
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()                             # destroy node after spin ends
        if rclpy.ok():
            rclpy.shutdown()                            # shutdown process


if __name__ == '__main__':
    main()