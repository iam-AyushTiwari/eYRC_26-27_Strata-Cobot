#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1A of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Functions:        detect_ores, depthimagecb, colorimagecb, caminfocb, _assign_ore_id,
#                   _update_ore_position, _broadcast_ores, process_image, main
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


##################### TASK CONSTANTS #######################

# Two ores of each type are spawned - six in all - told apart by an id of 1 or 2.
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

# The RealSense topics. The depth image is ALIGNED to the colour image.
color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'

# The parent frame every ore transform is published against.
base_frame = 'base_link'

hsv_ranges = {
    'azurite_ore':    ((110, 50, 50), (130, 255, 255)),
    'malachite_ore':  ((50, 50, 50),  (70, 255, 255)),
    'vanadinite_ore': ((5, 50, 50),   (25, 255, 255)),
}

ORE_HALF_HEIGHT = 0.025

MIN_ORE_AREA = 100          # contours smaller than this (pixels^2) are not ores
MATCH_PX = 80               # max pixel jump for a detection to keep its id
SAMPLES_PER_ORE = 10        # recent base_link samples medianed per ore
DEPTH_HALF_WINDOW = 4       # 9x9 depth window around the centre pixel

DETECTION_IMAGE = 'SC#1454_task1A_detection.png'   # written once, all six ores drawn
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

    ############ ADD YOUR CODE HERE ############

    # INSTRUCTIONS & HELP :

    #	->  Detect the ores by COLOUR, and return a center pixel and a type for each.
    #       ->  HINT: hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #                 mask = cv2.inRange(hsv, lower, upper)       # one pair per type
    #                 H is 0-179, S and V are 0-255. The frame is BGR, not RGB.

    #   ->  Read the bounds off a saved frame rather than copying them from a tutorial.

    #   ->  Clean the mask, and drop anything too small to be an ore.
    #       ->  HINT: cv2.morphologyEx (MORPH_OPEN, MORPH_CLOSE), cv2.contourArea

    #   ->  Reduce each region to one center pixel.
    #       ->  HINT: M  = cv2.moments(contour)                   # guard against m00 == 0
    #                 cX = int(M['m10'] / M['m00'])
    #                 cY = int(M['m01'] / M['m00'])

    #   ->  Draw your detections on the frame while you are developing.
    #       ->  HINT: cv2.circle, cv2.putText

    ############################################

    # blur so single noisy pixels do not survive the threshold, then BGR -> HSV
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for ore_type, (lower, upper) in hsv_ranges.items():
        mask = cv2.inRange(hsv_image,
                           np.array(lower, dtype=np.uint8),
                           np.array(upper, dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # only two ores of each type exist: keep the two largest regions, so a
        # stray blob can never take an id slot
        candidates = [c for c in contours if cv2.contourArea(c) > MIN_ORE_AREA]
        candidates.sort(key=cv2.contourArea, reverse=True)

        for cnt in candidates[:2]:
            M = cv2.moments(cnt)
            if M['m00'] == 0:                    # guard against zero division
                continue
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            center_ore_list.append((cX, cY))
            ore_type_list.append(ore_type)
            if contours_out is not None:
                contours_out.append(cnt)

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

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)

        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                         # depth image variable (from depthimagecb())
        self.cam_info = None                                                            # camera intrinsics variable (from caminfocb())

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Add any variable your detection needs to keep between frames.
        #       ->  HINT: The two ores of a type must keep their ids for the whole run, and
        #                 'detect_ores' returns them unordered.

        ############################################

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = None, None, None, None

        # Ore registry for assigning and maintaining IDs for each ore type
        self.ore_registry = {ot: [] for ot in ore_types}

        # Ore samples for storing the base_link positions of each ore
        self.ore_samples = {}

        self.detection_saved = False
        self.calibration_saved = False

        ############################################

    def depthimagecb(self, data):
        '''
        Description:    Callback function for the aligned depth camera topic.
                        Use this function to receive the depth image and convert it to a CV2 image.

        Args:
            data (Image):    Input depth image frame received from the aligned depth camera topic

        Returns: None : Convert the depth image msg to cv2 image for furhter processing
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Convert the ROS Image message to a CV2 image and store it.
        #       ->  HINT: self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

        #   ->  Get the units right. Print 'depth.dtype'.
        #       ->  HINT: 32FC1 is METRES; 16UC1 is MILLIMETRES, so divide by 1000.

        #   ->  Drop the pixels that carry no reading.
        #       ->  HINT: depth[np.isfinite(depth) & (depth > 0.0)]

        ############################################
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            # Convert depth to meters (16UC1 is mm -> divide by 1000)
            if depth_cv.dtype == np.uint16:
                self.depth_image = depth_cv.astype(np.float64) / 1000.0
            else:
                self.depth_image = depth_cv.astype(np.float64)
        except CvBridgeError as e:
            self.get_logger().error(f"Depth Image CvBridgeError: {e}")

    def colorimagecb(self, data):
        '''
        Description:    Callback function for the colour camera raw topic.
                        Use this function to receive the raw image and convert it to a CV2 image.

        Args:
            data (Image):    Input coloured raw image frame received from the image_raw camera topic

        Returns: None
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Convert the ROS Image message to a CV2 image and store it.
        #       ->  HINT: self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        ############################################
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Color Image CvBridgeError: {e}")



    def caminfocb(self, data):
        '''
        Description:    Callback function for the camera info topic.
                        Use this function to receive the camera's intrinsic parameters.

        Args:
            data (CameraInfo):    Camera calibration published by the camera

        Returns: None 
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Store the focal lengths and the principal point. Read them from the topic;
        #       never hard-code them.
        #       ->  HINT: 'k' is the pinhole matrix flattened row by row-
        #                     k = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

        ############################################
        self.cam_info = data
        # Camera intrinsic matrix K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = data.k[0]
        self.cx = data.k[2]
        self.fy = data.k[4]
        self.cy = data.k[5]

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
        known = self.ore_registry[ore_type]
        free = [i for i in range(len(known)) if (ore_type, i + 1) not in used]

        if free:
            best = min(free, key=lambda i: math.hypot(cX - known[i][0], cY - known[i][1]))
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

        Returns: None (leaves the samples untouched when there is no valid depth)
        '''
        rows, cols = self.depth_image.shape[:2]
        patch = self.depth_image[max(0, cY - DEPTH_HALF_WINDOW):min(rows, cY + DEPTH_HALF_WINDOW + 1),
                                 max(0, cX - DEPTH_HALF_WINDOW):min(cols, cX + DEPTH_HALF_WINDOW + 1)]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return

        z = float(np.median(valid))
        x = (cX - self.cx) * z / self.fx
        y = (cY - self.cy) * z / self.fy

        point_in_camera = PointStamped()
        point_in_camera.header.frame_id = optical_frame
        point_in_camera.header.stamp = self.get_clock().now().to_msg()
        point_in_camera.point.x = float(x)
        point_in_camera.point.y = float(y)
        point_in_camera.point.z = float(z)

        p = tf2_geometry_msgs.do_transform_point(point_in_camera, cam_to_base).point

        buf = self.ore_samples.setdefault(name, deque(maxlen=SAMPLES_PER_ORE))
        buf.append((p.x, p.y, p.z - ORE_HALF_HEIGHT))


    def _broadcast_ores(self):
        '''
        Description:    Broadcast every ore measured so far, every cycle, from the median
                        of its recent samples (steady against depth noise and dropouts).
        '''
        stamp = self.get_clock().now().to_msg()
        for name, buf in self.ore_samples.items():
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


    def process_image(self):
        '''
        Description:    Timer function used to detect the ores and publish a transform for each one on its estimated position.

        Args: None
        Returns: None
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #	->  Return early until both images and the camera info have arrived.

        #   ->  Get the ore centers and their types from 'detect_ores' defined above

        #   ->  Read the depth at each center pixel. Depth is ALIGNED to colour.
        #       ->  HINT: Take the MEDIAN of a small window, not the one pixel-
        #                     patch = self.depth_image[cY-4:cY+5, cX-4:cX+5]
        #                     z     = np.median(patch[np.isfinite(patch) & (patch > 0.0)])

        #   ->  Deproject the center pixel (u, v) and its depth z into a 3D point-
        #           x = (u - cx) * z / fx
        #           y = (v - cy) * z / fy
        #           z = z
        #       ->  HINT: That point is in the camera's OPTICAL frame - 'data.header.frame_id'.

        #   ->  Transform it into 'base_frame'-
        #           tf = self.tf_buffer.lookup_transform(
        #                    base_frame, <optical frame>, rclpy.time.Time())
        #           p  = do_transform_point(point_in_camera, tf)
        #       ->  HINT: PointStamped and tf2_geometry_msgs are not imported above, and the
        #                 lookup raises until the tree has filled in.

        #   ->  Correct for what you measured: the camera sees the ore's top face, and the
        #       ore is named by its middle.

        #   ->  Give each ore an id and keep it for the whole run.
        #       ->  HINT: Match a detection to the nearest ore of that type already named,
        #                 on the pixel rather than the depth.

        #   ->  Publish one transform per ore, using Geometry Message - TransformStamped
        #       Use the following frame_id-
        #           frame_id = 'base_link'
        #           child_frame_id = '<ore_type>_<id>'      Ex: azurite_ore_1, where azurite_ore
        #                                                   is one of 'ore_types' and the id is 1 or 2
        #       ->  HINT: t.header.stamp, t.header.frame_id, t.child_frame_id,
        #                 t.transform.translation.x/.y/.z, t.transform.rotation.w = 1.0,
        #                 then self.br.sendTransform(t). Every number must be a float.
        #       ->  NOTE: Only the translation is read; the names are matched exactly.

        #   ->  Broadcast every ore on every cycle, not once.

        #   ->  Show the frame with your detections drawn on it using 'cv2.imshow'.

        ############################################
         # Wait for all inputs to arrive
        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return
 
        if SAVE_CALIBRATION_FRAME and not self.calibration_saved:
            cv2.imwrite('calib_frame.png', self.cv_image)
            self.calibration_saved = True
 
        optical_frame = self.cam_info.header.frame_id
 
        try:
            cam_to_base = self.tf_buffer.lookup_transform(
                base_frame, optical_frame, rclpy.time.Time())
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)
            cam_to_base = None
 
        vis_image = self.cv_image.copy()        # annotate a copy, detect on the clean frame
        drawn = set()
 
        if cam_to_base is not None:
            contours = []
            center_ore_list, ore_type_list = detect_ores(self.cv_image, contours)
            used = set()
 
            for (cX, cY), ore_type, cnt in zip(center_ore_list, ore_type_list, contours):
                ore_id = self._assign_ore_id(ore_type, cX, cY, used)
                if ore_id is None:
                    continue
                used.add((ore_type, ore_id))
                name = f"{ore_type}_{ore_id}"
 
                self._update_ore_position(name, cX, cY, cam_to_base, optical_frame)
 
                # boundary + the exact transform name
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis_image, name, (x, max(y - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                drawn.add(name)
 
        # every ore measured so far, every cycle
        self._broadcast_ores()
 
        cv2.imshow("Ore Detector", vis_image)
        cv2.waitKey(1)
 
        # the submission image: all six ores, boundary and label on each
        if len(drawn) == 6 and not self.detection_saved:
            cv2.imwrite(DETECTION_IMAGE, vis_image)
            self.detection_saved = True
            self.get_logger().info(f"Saved {DETECTION_IMAGE}")


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
