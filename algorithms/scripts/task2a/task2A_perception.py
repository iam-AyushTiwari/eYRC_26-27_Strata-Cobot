#!/usr/bin/env python3



'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 2A of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         ore_detector.py
# Functions:        detect_ores, main
# Nodes:            Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/camera/color/image_raw, /camera/camera/aligned_depth_to_color/image_raw, /camera/camera/color/camera_info ]


################### IMPORT MODULES #######################

import rclpy
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
import tf2_geometry_msgs  # registers do_transform_point


# Two ores of each type are spawned - six in all - told apart by an id of 1 or 2.
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

# Four ores lies on the conveyor

# The RealSense topics. The depth image is ALIGNED to the colour image.
color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'

# The parent frame every ore transform is published against.
base_frame = 'base_link'


# HSV ranges for different ore types (tuned from mesh diffuse colors and Gazebo lighting)
hsv_ranges = {
    'azurite_ore':    ((95, 140, 60),  (118, 255, 255)),   # Centered on blue H=106
    'malachite_ore':  ((60, 140, 60),  (85, 255, 255)),    # Centered on green H=70
    'vanadinite_ore': ((0, 190, 80),   (15, 255, 255)),    # Centered on red-orange H=7, S>=190 rejects terrain (S~150)
}
# Secondary range for vanadinite red hue wrap near 180
vanadinite_wrap_range = ((170, 190, 80), (179, 255, 255))



# The ores are 101.6 x 101.6 x 76.2 mm (The Magnetic Tool). The camera measures the TOP face, and the published frame sits TF_BELOW_TOP below it: 0.0381 m (half the height) puts the frame at the ore's middle, 0.0 puts it on the top face. 

TF_BELOW_TOP = 0.0381
ORE_HALF_HEIGHT = 0.0381    # half the ore height (76.2 mm / 2); offsets z from top-face to centre
EXPECTED_ORES = 4  # Four ores lies on the conveyor

MIN_ORE_AREA = 100          # contours smaller than this (pixels^2) are not ores
MAX_ORE_AREA = 3000         # contours larger than this (pixels^2) are terrain, not ores
MATCH_PX = 80               # max pixel jump for a detection to keep its id
SAMPLES_PER_ORE = 10        # recent base_link samples medianed per ore
DEPTH_HALF_WINDOW = 4       # 9x9 depth window around the centre pixel

#Once every ore has a full, steady buffer , stop detecting and keep brodcasting those positions. A carried ore moves, and for a kind with a single ore it would otherwise be registered as the next ore.  Setting flase if the conveyor turns out to move 

FREEZE_WHEN_STABLE = True
FREEZE_SPREAD = 0.010       # metres; the buffer must agree to within this on every axis
 
SHOW_WINDOW = True     

SAVE_CALIBRATION_FRAME = False                      # True: save one raw frame to sample HSV


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

    ############ Function VARIABLES ############

    # ->  You can remove these variables if needed. These are just for suggestions to let you get started

    center_ore_list = []
    ore_type_list = []

    # blur so single noisy pixels do not survive the threshold, then BGR -> HSV
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0) # blur by 5x5 kernel to reduce noise
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV) # Convert BGR to HSV

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # Rectangular kernel of 3x3

    # threshold on HSV ranges and then clean and reduce to one center per ore
    for ore_type, (lower, upper) in hsv_ranges.items():
        mask = cv2.inRange(hsv_image,
                           np.array(lower, dtype=np.uint8),
                           np.array(upper, dtype=np.uint8))

        # Red hue wraps around 0 and 179: combine with wrap range for vanadinite
        if ore_type == 'vanadinite_ore':
            mask_wrap = cv2.inRange(hsv_image,
                                    np.array(vanadinite_wrap_range[0], dtype=np.uint8),
                                    np.array(vanadinite_wrap_range[1], dtype=np.uint8))
            mask = cv2.bitwise_or(mask, mask_wrap)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Discard terrain (> MAX_ORE_AREA), tiny noise (< MIN_ORE_AREA), and extreme aspect ratios
        candidates = []
        for c in contours:
            a = cv2.contourArea(c)
            if not (MIN_ORE_AREA <= a <= MAX_ORE_AREA):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if 0.4 <= float(w) / h <= 2.5:
                candidates.append(c)

        candidates.sort(key=cv2.contourArea, reverse=True)

        # keep only the top 2 candidates, to avoid noise
        for cnt in candidates[:2]:
            M = cv2.moments(cnt)
            if M['m00'] == 0: # guard against zero division
                continue
            cX = int(M['m10'] / M['m00'])   # center pixel coordinates
            cY = int(M['m01'] / M['m00'])

            center_ore_list.append((cX, cY)) # append center pixel coordinates
            ore_type_list.append(ore_type)   # append ore type

    ############################################

    return center_ore_list, ore_type_list


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

        super().__init__('ore_tf_publisher')                                            # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.caminfocb, 10)
        

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2 # rate of time to process image (seconds)
        self.bridge = CvBridge()    # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer() # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.br = tf2_ros.TransformBroadcaster(self) # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image) # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)

        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                         # depth image variable (from depthimagecb())
        self.cam_info = None                                                            # camera intrinsics variable (from caminfocb())


        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = None, None, None, None

        # Ore registry for assigning and maintaining IDs for each ore type
        self.ore_registry = {ot: [] for ot in ore_types}

        # Ore samples for storing the base_link positions of each ore
        self.ore_samples = {}

        self.frozen = False # variable to check if the ore detection is frozen

        self.detection_saved = False # flag to ensure the detection image is saved only once
        self.calibration_saved = False # flag to ensure the calibration frame is saved only once

        ############################################

    def depthimagecb(self, data):
        '''
        Description:    Callback function for the aligned depth camera topic.
                        Use this function to receive the depth image and convert it to a CV2 image.

        Args:
            data (Image):    Input depth image frame received from the aligned depth camera topic

        Returns: 
        '''


        # try-except block for CvBridgeError
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # convert ROS Image msg to CV2 image
            # Convert depth to meters (16UC1 is mm -> divide by 1000)
            if depth_cv.dtype == np.uint16:
                self.depth_image = depth_cv.astype(np.float64) / 1000.0
            else:
                self.depth_image = depth_cv.astype(np.float64)                                 # store depth image in float64 format
        except CvBridgeError as e:
            self.get_logger().error(f"Depth Image CvBridgeError: {e}")

    def colorimagecb(self, data):
        '''
        Description:    Callback function for the colour camera raw topic.
                        Use this function to receive the raw image and convert it to a CV2 image.

        Args:
            data (Image):    Input coloured raw image frame received from the image_raw camera topic

        Returns: 
        '''


        # try-except block for CvBridgeError
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')   # convert ROS Image msg to CV2 image
        except CvBridgeError as e:
            self.get_logger().error(f"Color Image CvBridgeError: {e}")



    def caminfocb(self, data):
        '''
        Description:    Callback function for the camera info topic.
                        Use this function to receive the camera's intrinsic parameters.

        Args:
            data (CameraInfo):    Camera calibration published by the camera

        Returns: 
        '''


        self.cam_info = data
        # Camera intrinsic matrix K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = data.k[0]   # focal length in x direction
        self.cx = data.k[2]   # principal point in x direction
        self.fy = data.k[4]   # focal length in y direction
        self.cy = data.k[5]   # principal point in y direction

        ############################################


    def _assign_ore_id(self, ore_type, cX, cY, used):
        '''
        Description:    Match a detection to an already-named ore of the same type by
                        pixel distance, or name it if it is new.

        Args:
            ore_type (str):   one of 'ore_types'
            cX, cY   (int):   centre pixel of the detection
            used     (set):   (ore_type, id) pairs already claimed this cycle, so two
                              detections can never publish the same name

        Returns:
            1 or 2, or None if the detection cannot be one of the two ores of this type
        '''

        known = self.ore_registry[ore_type] # get known ores of the same type
        free = [i for i in range(len(known)) if (ore_type, i + 1) not in used] # get free ores of the same type

        # if there are free ores, find the closest one
        if free:
            best = min(free, key=lambda i: math.hypot(cX - known[i][0], cY - known[i][1])) # find the closest ore
            if math.hypot(cX - known[best][0], cY - known[best][1]) <= MATCH_PX:
                known[best] = (cX, cY)
                return best + 1

        if len(known) < 2:
            known.append((cX, cY))
            return len(known)

        return None


    def _update_ore_position(self, name, cX, cY, cam_to_base, optical_frame):
        '''
        Description:    Read depth at the ore centre, deproject it, move it into base_frame,
                        and add it to the ore's recent samples.

        Args:
            name (str):          name of the ore, e.g. 'azurite_ore_1'
            cX, cY (int):        centre pixel of the detection
            cam_to_base (TransformStamped):   transform from camera to base frame
            optical_frame (str):   frame id of the optical frame
        Returns
        '''

        rows, cols = self.depth_image.shape[:2] # get the shape of the depth image
        patch = self.depth_image[max(0, cY - DEPTH_HALF_WINDOW):min(rows, cY + DEPTH_HALF_WINDOW + 1),
                                 max(0, cX - DEPTH_HALF_WINDOW):min(cols, cX + DEPTH_HALF_WINDOW + 1)] 
        valid = patch[np.isfinite(patch) & (patch > 0.0)] # get valid depth values
        
        if valid.size == 0:
            return

        z = float(np.median(valid))     # calculate the median depth
        x = (cX - self.cx) * z / self.fx  # convert pixel coordinates to Cartesian coordinates
        y = (cY - self.cy) * z / self.fy

        point_in_camera = PointStamped()             # create a PointStamped object
        point_in_camera.header.frame_id = optical_frame        # set the frame id
        point_in_camera.header.stamp = self.get_clock().now().to_msg()   # set the time stamp
        point_in_camera.point.x = float(x)
        point_in_camera.point.y = float(y)
        point_in_camera.point.z = float(z)

        p = tf2_geometry_msgs.do_transform_point(point_in_camera, cam_to_base).point   # transform the point to the base frame

        buf = self.ore_samples.setdefault(name, deque(maxlen=SAMPLES_PER_ORE))
        buf.append((p.x, p.y, p.z - ORE_HALF_HEIGHT))


    def _broadcast_ores(self):
        '''
        Description:    Broadcast every ore measured so far, every cycle, from the median
                        of its recent samples (steady against depth noise and dropouts).
        Returns:
        '''
        stamp = self.get_clock().now().to_msg()     # get the current time
        for name, buf in self.ore_samples.items():  # iterate through the ore samples
            x, y, z = np.median(np.array(buf), axis=0)  # calculate the median of the samples

            t = TransformStamped()                  # create a TransformStamped object
            t.header.stamp = stamp                  # set the time stamp
            t.header.frame_id = base_frame          # set the frame id
            t.child_frame_id = name                 # set the child frame id
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
 
        if len(self.ore_samples) > EXPECTED_ORES:
            self.get_logger().warn(
                f"{len(self.ore_samples)} ore names but only {EXPECTED_ORES} ores exist: {sorted(self.ore_samples)}",
                throttle_duration_sec=5.0)
            return False
 
        if len(self.ore_samples) != EXPECTED_ORES:
            return False
 
        for buf in self.ore_samples.values():
            if len(buf) < SAMPLES_PER_ORE:
                return False
            if np.ptp(np.array(buf), axis=0).max() > FREEZE_SPREAD:
                return False
        return True
 
    def process_image(self):
        '''
        Description:    Timer function used to detect the ores and publish a transform for each one on its estimated position.

        Args: 
        Returns:
        '''
        
      # Wait for all inputs to arrive
        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return

        # Save calibration frame if requested and not already saved
        if SAVE_CALIBRATION_FRAME and not self.calibration_saved:
            cv2.imwrite('calib_frame.png', self.cv_image)
            self.calibration_saved = True
 
        optical_frame = self.cam_info.header.frame_id   # get the optical frame from the camera info
 
        try:
            cam_to_base = self.tf_buffer.lookup_transform(
                base_frame, optical_frame, rclpy.time.Time())   # lookup the transform from the optical frame to the base frame
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)
            cam_to_base = None
 
        vis_image = self.cv_image.copy()      # annotate a copy, detect on the clean frame

        # Check if we should freeze positions because they are stable
        if not self.frozen and self._ready_to_freeze():
            self.frozen = True
            self.get_logger().info("All 4 ores stable! Freezing positions for gripping.")

        # Detect and update positions only if not frozen
        if not self.frozen and cam_to_base is not None:
            center_ore_list, ore_type_list = detect_ores(self.cv_image)
            used = set()
 
            for (cX, cY), ore_type in zip(center_ore_list, ore_type_list):
                ore_id = self._assign_ore_id(ore_type, cX, cY, used)
                if ore_id is None:
                    continue
                used.add((ore_type, ore_id))
                name = f"{ore_type}_{ore_id}"
 
                self._update_ore_position(name, cX, cY, cam_to_base, optical_frame)
 
                # boundary + the exact transform name
                box_size = 15
                x_axis, y_axis, width, height = cX - box_size, cY - box_size, box_size * 2, box_size * 2
                cv2.rectangle(vis_image, (x_axis, y_axis), (x_axis + width, y_axis + height), (0, 255, 0), 2)
                cv2.putText(vis_image, name, (x_axis, max(y_axis - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
 
        # every ore measured so far, every cycle
        self._broadcast_ores()
 
        if SHOW_WINDOW:
            cv2.imshow("Ore Detector", vis_image)
            cv2.waitKey(1)

##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the ore_tf class to perform its task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('ore_tf_process')                      # creating ROS node

    node.get_logger().info('Node created: Ore tf process')          # logging information

    ore_tf_class = ore_tf()                                         # creating a new object for class 'ore_tf'

    rclpy.spin(ore_tf_class)                                        # spining on the object to make it alive in ROS 2 DDS

    ore_tf_class.destroy_node()                                     # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':

    main()
