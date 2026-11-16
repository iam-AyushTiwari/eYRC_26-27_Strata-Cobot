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
# Author List:		Aryan Joshi, Ayush Tiwari
# Filename:		    task3B_planning.py
# Functions:
#			        world_to_cell, cell_to_world, build_clearance_map, astar, line_is_clear, shorten_path, resample_path, path_length, estimate_map_offset, shift_grid, main

# Nodes:		    Publishing Topics  - [ /ebot_path, /ebot_at_arm ]
#                   Subscribing Topics - [ /map, /odom, /scan, /package_picked, /tf ]
#                   Service Clients    - [ /spawn_ore_package ]


################### IMPORT MODULES #######################

import rclpy
import rclpy.executors
import sys
import math
import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_srvs.srv import Trigger
from std_msgs.msg import Bool



##################### TASK CONSTANTS #######################

# The map is published ONCE, as the world loads, and held for late subscribers.
map_topic = '/map'

# Your follower reads the path from here, with the same held-message QoS.
path_topic = '/ebot_path'

odom_topic = '/odom'
scan_topic = '/scan'
laser_frame = 'laser'
spawn_service = '/spawn_ore_package'

# The handover with the arm, on two held topics of our own:
#   /ebot_at_arm     True while the eBot stands at the stop pose with the package on its deck
#   /package_picked  True once the arm has lifted the package clear of the deck
at_arm_topic = '/ebot_at_arm'
package_picked_topic = '/package_picked'

# Every pose in the path, and the path itself, is in this frame.
map_frame = 'map'

# The three points of the lap: (x, y) in metres and yaw in radians, in the map frame.
# At the ore drop pose either yaw, 0.0 or 3.14, is valid. arm_pose is the stop pose beside
# the arm (base at world (-0.597, 1.865)) chosen so the package on the deck is in reach.
home_pose = (0.0, 0.0, 0.0)
ore_drop_pose = (6.0, 1.9, math.pi)   # facing back along leg 2, so the lidar sees that route while the eBot waits
arm_pose = (0.0, 1.9, math.pi)

# The lap, as (start, end) of each step, in the order they are driven.
steps = [
    (home_pose, ore_drop_pose),     # step 1: then request the ore package
    (ore_drop_pose, arm_pose),      # step 2
    (arm_pose, home_pose),          # step 3
]

# Robot radius 
ROBOT_RADIUS = 0.40


# Extra room on top of robot radius for tracking error and wheel slippage
SAFETY_MARGIN = 0.12 

# Beyond the hard margin, cells are merely dearer, so the path drifts to the middle of a gap
SOFT_BAND = 0.25
SOFT_WEIGHT = 3.0

# Cells this close to a step's start or end are never blocked, so a rock near a pose cannot 
# make the search impossible.
ENDPOINT_CLEAR_RADIUS = 0.30

RESAMPLE_SPACING = 0.25         # metres between published poses
OCCUPIED_THRESHOLD = 50         # a cell at or above this is a rock (free is 0, occupied is 100)


# Standing still, from /odom.
STILL_LINEAR = 0.02             # m/s
STILL_ANGULAR = 0.02            # rad/s
PLAN_STILL_S = 1.5              # seconds still before a path is published (the lidar counts hits meanwhile)

# A step is finished when the eBot is this close to its end pose and has stood still a while. (according to task guide it is 0.3m and 0.15 rad)
ARRIVE_POS_TOL = 0.30           # Position tolerance (m)
ARRIVE_YAW_TOL = 0.15           # Yaw tolerance (rad)
ARRIVE_STILL_S = 1.0            # (seconds ebot stands at the final point for it to be detected as 'arrived'.)

PACKAGE_RETRY_S = 1.5           # seconds between /spawn_ore_package attempts
MAX_PACKAGE_TRIES = 5
PACKAGE_SETTLE_S = 2.0          # let the package land before moving off
STEP_TIMEOUT_S = 180.0          # give up waiting for a step after this long
HANDOVER_TIMEOUT_S = 400.0      # leave the arm anyway if the package is never taken

# Odometry is noisy in Task 3B, so "standing still" is judged on smoothed velocities.
VELOCITY_FILTER_ALPHA = 0.2     # weight of the newest reading in the running average

# /map may not match the world in Task 3B. While the eBot stands still, the lidar's hits are
# counted on a coarse grid; a cell hit often enough is a real obstacle and is planned around
# as well as the map's rocks. The hits are also matched against /map to estimate how far the
# rocks stand from where the map shows them.
OBS_CELL = 0.05                 # m, side of one cell of the lidar hit grid
OBS_MIN_HITS = 3                # hits before a cell counts as an obstacle
SCAN_MAX_RANGE = 5.0            # m, hits further than this are too uncertain to use
OFFSET_SEARCH = 0.50            # m, largest rock offset searched for, in x and in y
OFFSET_STEP = 0.025             # m, step of that search
OFFSET_CLIP = 0.30              # m, a hit further than this from any rock adds only this to the match cost
OFFSET_MIN_POINTS = 40          # obstacle cells needed before an offset is estimated
OFFSET_MIN_GAIN = 0.6           # the best offset must cost below this share of no offset at all

SQRT2 = math.sqrt(2.0)

MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]



##################### FUNCTION DEFINITIONS #######################
 
def world_to_cell(x, y, origin, resolution):
    '''
    Description:    Convert a map position to the grid cell that contains it.
 
    Args:
        x, y (float):        position in the map frame, in metres
        origin (tuple):      (x, y) of the corner of cell (0, 0)
        resolution (float):  side of one cell, in metres
 
    Returns:
        (row, col) (int, int)
    '''
    return (int(math.floor((y - origin[1]) / resolution)),
            int(math.floor((x - origin[0]) / resolution)))


def cell_to_world(row, col, origin, resolution):
    '''
    Description:    Convert a grid cell to the map position of its CENTRE.
 
    Args:
        row, col (int):      the cell
        origin (tuple):      (x, y) of the corner of cell (0, 0)
        resolution (float):  side of one cell, in metres
 
    Returns:
        (x, y) (float, float)
    '''
    return (origin[0] + (col + 0.5) * resolution,
            origin[1] + (row + 0.5) * resolution)
 

def build_clearance_map(grid, resolution):
    '''
    Description:    For every cell, the distance in metres from the cell to the nearest rock.
 
    Args:
        grid (ndarray):      occupancy values, shape (height, width)
        resolution (float):  side of one cell, in metres
 
    Returns:
        clearance (ndarray): metres, inf everywhere if there are no rocks
    '''
    # check if there are any occupied cells
    occupied = grid >= OCCUPIED_THRESHOLD
    # if there are no occupied cells, return infinity
    if not occupied.any():
        return np.full(grid.shape, np.inf)
    # distances are between cell centres; half a cell off the rock's true surface
    return distance_transform_edt(~occupied) * resolution - 0.5 * resolution
 

def astar(blocked, penalty, start, goal):
    '''
    Description:    A* over the grid, 8-connected, with the octile distance as heuristic.
 
    Args:
        blocked (ndarray):   True where the eBot's centre may not go
        penalty (ndarray):   extra cost per unit step into each cell (0 = none)
        start (tuple):       (row, col)
        goal (tuple):        (row, col)
 
    Returns:
        list:                [(row, col), ...] from start to goal, or None if there is no route
    '''
    height, width = blocked.shape
    blk = blocked.ravel().tolist()
    pen = penalty.ravel().tolist()
    size = height * width
    g = [math.inf] * size
    parent = [-1] * size
    closed = [False] * size
 
    s = start[0] * width + start[1]
    t = goal[0] * width + goal[1]
    goal_row, goal_col = goal
 
    def heuristic(index):
        '''
        Description:    Octile distance from a cell to the goal, a lower bound on the cost left.

        Args:
            index (int):        flat index of the cell, row * width + col

        Returns:
            distance (float):   estimated cost from the cell to the goal, in cells
        '''
        row, col = divmod(index, width)
        dx = abs(col - goal_col)
        dy = abs(row - goal_row)
        return dx + dy + (SQRT2 - 2.0) * min(dx, dy)
 
    g[s] = 0.0 # g stores the cost from the start node to the current node
    heap = [(heuristic(s), s)] # heap stores the nodes to be visited
 
    while heap:
        _, i = heapq.heappop(heap)
        if closed[i]:
            continue                      # a stale duplicate
        closed[i] = True
        if i == t:
            break
        row, col = divmod(i, width)
        for d_row, d_col, step in MOVES:
            n_row = row + d_row
            n_col = col + d_col
            # check if the new cell is within the grid
            if n_row < 0 or n_row >= height or n_col < 0 or n_col >= width:
                continue
            j = n_row * width + n_col
            # check if the new cell is already closed or blocked
            if closed[j] or blk[j]:
                continue
            # never cut a corner between two cells, one of which is blocked
            if d_row and d_col and (blk[(row + d_row) * width + col] or blk[row * width + n_col]):
                continue
            new_g = g[i] + step * (1.0 + pen[j])
            # if the new cost is less than the current cost
            if new_g < g[j]:
                g[j] = new_g
                parent[j] = i
                heapq.heappush(heap, (new_g + heuristic(j), j))
    # if the goal is not reached
    if not closed[t]:
        return None
 
    cells = [] # list to store cells of the path
    i = t
    # reconstruct the path from goal to start
    while i != -1:
        cells.append(divmod(i, width))
        i = parent[i]
    # reverse the path to get the path from start to goal
    cells.reverse()
    return cells

def line_is_clear(blocked, a, b):
    '''
    Description:    Whether the straight line between two cells crosses a blocked cell.
 
    Args:
        blocked (ndarray):   True where the eBot's centre may not go
        a, b (tuple):        (row, col) of both ends
 
    Returns:
        clear (bool)
    '''
    # count is the number of cells to check on the line
    count = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    rows = np.rint(np.linspace(a[0], b[0], count)).astype(int)
    cols = np.rint(np.linspace(a[1], b[1], count)).astype(int)
    # check if any of the cells on the line are blocked
    return not blocked[rows, cols].any()
 
 
def shorten_path(cells, blocked):
    '''
    Description:    Cut a grid path down to the corners that matter, by jumping to the
                    farthest cell that can be seen in a straight clear line.
 
    Args:
        cells (list):        [(row, col), ...] from astar()
        blocked (ndarray):   True where the eBot's centre may not go
 
    Returns:
        list:                the shortened [(row, col), ...]
    '''
    shortened = [cells[0]] # start with the first cell
    i = 0 # index of the current cell
    last = len(cells) - 1 # index of the last cell
    # go through each cell from start to second-to-last
    while i < last:
        j = last
        # go backwards from last cell to current cell
        while j > i + 1 and not line_is_clear(blocked, cells[i], cells[j]):
            j -= 1
        shortened.append(cells[j])
        i = j
    return shortened
 
 
def resample_path(points, spacing):
    '''
    Description:    Add points along every straight segment so none is longer than 'spacing'.
                    Every original corner is kept.
 
    Args:
        points (list):       [(x, y), ...]
        spacing (float):     longest allowed gap, in metres
 
    Returns:
        list:                [(x, y), ...]
    '''
    result = [points[0]] # start from the first point
    # go through each pair of points
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        pieces = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / spacing)))
        for k in range(1, pieces + 1):
            f = k / pieces
            result.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
    return result
 
 
def estimate_map_offset(points, clearance, origin, resolution):
    '''
    Description:    Finds the shift (dx, dy) that best lays the lidar's obstacle points on the
                    map's rocks, by trying every shift on a grid.

    Args:
        points (ndarray):    (N, 2) obstacle positions seen by the lidar, in the map frame
        clearance (ndarray): metres from each map cell to the nearest rock, from build_clearance_map()
        origin (tuple):      (x, y) of the corner of cell (0, 0)
        resolution (float):  side of one cell, in metres

    Returns:
        offset (tuple):      (dx, dy) in metres, the world's rocks stand at map + offset; (0, 0) if unsure
        gain (float):        cost at the best offset as a share of the cost at no offset (lower is surer)
    '''
    if len(points) < OFFSET_MIN_POINTS:
        return (0.0, 0.0), 1.0
    height, width = clearance.shape
    steps = np.arange(-OFFSET_SEARCH, OFFSET_SEARCH + 1e-9, OFFSET_STEP)

    def cost(dx, dy):
        '''
        Description:    How badly the lidar's points sit on the map's rocks once moved back by (dx, dy).

        Args:
            dx (float):     offset tried along x, in metres
            dy (float):     offset tried along y, in metres

        Returns:
            cost (float):   mean clipped distance from each point to the nearest rock, in metres
        '''
        cols = np.floor((points[:, 0] - dx - origin[0]) / resolution).astype(int)
        rows = np.floor((points[:, 1] - dy - origin[1]) / resolution).astype(int)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        values = np.full(len(points), OFFSET_CLIP)
        values[inside] = np.clip(clearance[rows[inside], cols[inside]], 0.0, OFFSET_CLIP)
        return values.mean()

    base = cost(0.0, 0.0)
    best = min(((cost(dx, dy), dx, dy) for dx in steps for dy in steps), key=lambda c: (c[0], abs(c[1]) + abs(c[2])))
    gain = best[0] / base if base > 0 else 1.0
    if gain > OFFSET_MIN_GAIN:
        return (0.0, 0.0), gain
    return (float(best[1]), float(best[2])), gain


def shift_grid(grid, d_rows, d_cols, fill):
    '''
    Description:    Moves a grid by whole cells, filling the cells uncovered with 'fill'.

    Args:
        grid (ndarray):      the grid to move
        d_rows (int):        cells to move it up the rows (positive is +y)
        d_cols (int):        cells to move it along the columns (positive is +x)
        fill:                value for the uncovered cells

    Returns:
        shifted (ndarray):   the moved grid, the same shape as 'grid'
    '''
    shifted = np.full_like(grid, fill)
    height, width = grid.shape
    src_rows = slice(max(0, -d_rows), min(height, height - d_rows))
    dst_rows = slice(max(0, d_rows), min(height, height + d_rows))
    src_cols = slice(max(0, -d_cols), min(width, width - d_cols))
    dst_cols = slice(max(0, d_cols), min(width, width + d_cols))
    shifted[dst_rows, dst_cols] = grid[src_rows, src_cols]
    return shifted


def path_length(points):
    '''
    Description:    Length of a polyline, in metres.
 
    Args:
        points (list):       [(x, y), ...]
 
    Returns:
        length (float)
    '''
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]))
 
 
 
##################### CLASS DEFINITION #######################

class ebot_planner(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to read the map, plan a path for each
                    step of the lap, publish it on /ebot_path, and request the ore package
                    at the end of step 1.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_planner
        '''

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
        super().__init__(                                                               # registering node
            'ebot_planner_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # A held message: kept by the publisher and handed to anyone who subscribes later.
        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        ############ Topic SUBSCRIPTIONS ############

        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self.mapcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scancb, qos_profile_sensor_data)
        self.tf_buffer = tf2_ros.Buffer()                                               # where the laser is, for placing its hits on the map
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.picked_sub = self.create_subscription(Bool, package_picked_topic, self.packagepickedcb, latched)

        ############ Topic PUBLISHERS ############

        self.path_pub = self.create_publisher(Path, path_topic, latched)                # your follower drives what you publish here
        self.at_arm_pub = self.create_publisher(Bool, at_arm_topic, latched)            # tells the arm the package is waiting beside it

        ############ Service CLIENTS ############

        self.spawn_client = self.create_client(Trigger, spawn_service)                  # drops the ore package onto the eBot

        ############ Constructor VARIABLES/OBJECTS ############

        plan_rate = 0.1                                                                 # rate of time to run one planning cycle (seconds)
        self.timer = self.create_timer(plan_rate, self.process_planning)                # creating a timer based function which gets called on every 0.1 seconds (as defined by 'plan_rate' variable)

        self.grid = None                                                                # the map (from mapcb())
        self.odom = None                                                                # where the base is (from odomcb())
        self.step = 0                                         


        # The map, once read
        self.resolution = None                    # metres per cell
        self.origin = None                        # (x, y) of the corner of cell (0, 0)
        self.clearance = None                     # metres from each cell to the nearest rock
 
        # Where the lap has got to: PLAN => DRIVE => (PACKAGE => SETTLE after step 1) => PLAN ... => DONE
        self.phase = 'PLAN'
        self.phase_t0 = 0.0
        self.still_since = None                 # when the eBot last stopped moving
 
        # The ore package request
        self.package_pending = False
        self.package_ok = False
        self.package_tries = 0
        self.last_package_try = -math.inf

        # What the lidar has seen, and how far the rocks stand from the map
        self.obs_hits = None                    # hit count per OBS_CELL cell (built in mapcb())
        self.map_offset = (0.0, 0.0)            # estimated (dx, dy) from the map's rocks to the world's
        self.velocity = (0.0, 0.0)              # smoothed (linear, angular) velocity, from odomcb()

        # The handover with the arm
        self.package_picked = False             # set once the arm reports the package off the deck (from packagepickedcb())
        self.publish_at_arm(False)

    def packagepickedcb(self, data):
        '''
        Description:    Callback function for /package_picked, published by the arm node.

        Args:
            data (Bool):    True once the arm has lifted the package off the deck

        Returns:
        '''
        if data.data and not self.package_picked:
            self.get_logger().info("Arm reports the package is off the deck")
        self.package_picked = bool(data.data)

    def publish_at_arm(self, value):
        '''
        Description:    Publishes whether the eBot is waiting beside the arm with the package.

        Args:
            value (bool):    True while the eBot stands at the stop pose with the package

        Returns:
        '''
        msg = Bool()
        msg.data = bool(value)
        self.at_arm_pub.publish(msg)

    def mapcb(self, data):
        '''
        Description:    Callback function for the map topic.
                        Use this function to receive the occupancy grid of the arena.

        Args:
            data (OccupancyGrid):    The arena, one value per cell

        Returns:
        '''

        info = data.info

        # row-major, so reshaping to (height, width) gives grid[row, col]
        self.grid = np.array(data.data, dtype=np.int16).reshape(info.height, info.width)

        self.resolution = float(info.resolution) # converting from 0.05 m/cell to m
        
        self.origin = (float(info.origin.position.x), float(info.origin.position.y)) # origin of the map

        self.clearance = build_clearance_map(self.grid, self.resolution) # builds a map of clearance for each cell
        self.obs_scale = max(1, int(round(OBS_CELL / self.resolution)))  # map cells per lidar-grid cell
        self.obs_hits = np.zeros((-(-info.height // self.obs_scale), -(-info.width // self.obs_scale)), dtype=np.int32)
 
        self.get_logger().info(
            f"Map received: {info.width} x {info.height} cells at {self.resolution:.3f} m, "
            f"origin ({self.origin[0]:.2f}, {self.origin[1]:.2f}), "
            f"{int((self.grid >= OCCUPIED_THRESHOLD).sum())} occupied cells")

    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.
                        Use this function to receive where the base currently is.

        Args:
            data (Odometry):    Pose and velocity of the base

        Returns:
        '''
        
        position = data.pose.pose.position # position of the base in (x, y)
        orientation = data.pose.pose.orientation # orientation of the base in quaternion format

        yaw = math.atan2(2.0 * (orientation.w * orientation.z + orientation.x * orientation.y), 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)) # converting quaternion to yaw

        # smoothing the velocities, which carry noise in Task 3B, before judging standing still
        alpha = VELOCITY_FILTER_ALPHA
        self.velocity = (alpha * data.twist.twist.linear.x + (1.0 - alpha) * self.velocity[0],
                         alpha * data.twist.twist.angular.z + (1.0 - alpha) * self.velocity[1])

        self.odom = (position.x, position.y, yaw, self.velocity[0], self.velocity[1]) # storing the odom values in self.odom

    def scancb(self, data):
        '''
        Description:    Callback function for the lidar. While the eBot stands still, counts
                        each hit on the coarse lidar grid, in the map frame.

        Args:
            data (LaserScan):    one sweep of the 2D lidar

        Returns:
        '''
        if self.obs_hits is None or self.odom is None:
            return
        if abs(self.velocity[0]) > STILL_LINEAR or abs(self.velocity[1]) > STILL_ANGULAR:
            return                                  # a sweep taken on the move is smeared by the motion
        try:
            laser = self.tf_buffer.lookup_transform(map_frame, laser_frame, rclpy.time.Time()).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        q = laser.rotation
        laser_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        ranges = np.array(data.ranges, dtype=float)
        angles = data.angle_min + data.angle_increment * np.arange(len(ranges))
        good = np.isfinite(ranges) & (ranges > data.range_min) & (ranges < min(data.range_max, SCAN_MAX_RANGE))
        xs = laser.translation.x + ranges[good] * np.cos(laser_yaw + angles[good])
        ys = laser.translation.y + ranges[good] * np.sin(laser_yaw + angles[good])

        cell = self.resolution * self.obs_scale
        rows = np.floor((ys - self.origin[1]) / cell).astype(int)
        cols = np.floor((xs - self.origin[0]) / cell).astype(int)
        inside = (rows >= 0) & (rows < self.obs_hits.shape[0]) & (cols >= 0) & (cols < self.obs_hits.shape[1])
        np.add.at(self.obs_hits, (rows[inside], cols[inside]), 1)

    def observed_points(self):
        '''
        Description:    Centres of the lidar-grid cells hit often enough to be real obstacles.

        Args:
        Returns:
            points (ndarray):    (N, 2) positions in the map frame
        '''
        rows, cols = np.nonzero(self.obs_hits >= OBS_MIN_HITS)
        cell = self.resolution * self.obs_scale
        return np.stack([self.origin[0] + (cols + 0.5) * cell, self.origin[1] + (rows + 0.5) * cell], axis=1)

    def planning_clearances(self):
        '''
        Description:    The clearance maps to plan on, most cautious first: the map's rocks
                        with what the lidar has seen and the rocks moved by the estimated
                        offset, then with only what the lidar has seen, then the map alone.

        Args:
        Returns:
            clearances (list):   [(name, clearance ndarray), ...] in the order to try them
        '''
        options = []
        observed = self.obs_hits >= OBS_MIN_HITS
        if observed.any():
            fine = np.kron(observed, np.ones((self.obs_scale, self.obs_scale), dtype=bool))
            fine = fine[:self.grid.shape[0], :self.grid.shape[1]]
            seen = np.where(fine, 100, 0).astype(np.int16)
            seen_clearance = np.minimum(self.clearance, build_clearance_map(seen, self.resolution))

            self.map_offset, gain = estimate_map_offset(
                self.observed_points(), self.clearance, self.origin, self.resolution)
            if self.map_offset != (0.0, 0.0):
                self.get_logger().info(
                    f"Rocks stand about ({self.map_offset[0]:+.3f}, {self.map_offset[1]:+.3f}) m from the map "
                    f"(match cost {gain:.2f} of none)")
                moved = shift_grid(self.grid, int(round(self.map_offset[1] / self.resolution)),
                                   int(round(self.map_offset[0] / self.resolution)), 0)
                options.append(('map, lidar and moved rocks',
                                np.minimum(seen_clearance, build_clearance_map(moved, self.resolution))))
            options.append(('map and lidar', seen_clearance))
        options.append(('map only', self.clearance))
        return options

    def now_s(self):
        '''
        Description:    The node clock (the simulation clock) in seconds.
 
        Args:
        Returns:
            seconds (float)
        '''
        return self.get_clock().now().nanoseconds / 1e9 # returns time in seconds


    def plan_path(self, start, end):
        '''
        Description:    Plans a path from 'start' to 'end' that stays clear of every rock.

        Args:
            start (tuple):    (x, y, yaw) where the step begins, in the map frame
            end (tuple):      (x, y, yaw) where the step ends, in the map frame

        Returns:
            list:             [(x, y), ...] from 'start' to 'end', or None if no path was found
        '''

        points = None
        map_clearance = self.clearance

        # Trying the most cautious obstacle picture first, then with different margins.
        # Every picture includes the map's rocks, so no path ever crosses an occupied map cell.
        try:
            for name, clearance in self.planning_clearances():
                self.clearance = clearance
                for margin, band in ((SAFETY_MARGIN, SOFT_BAND), (SAFETY_MARGIN * 0.5, 0.0), (0.0, 0.0)):
                    points = self._plan(start, end, ROBOT_RADIUS + margin, band)
                    if points is not None:
                        if margin < SAFETY_MARGIN:
                            self.get_logger().warn(
                                f"No path with the full {SAFETY_MARGIN:.2f} m margin; planned with {margin:.2f} m")
                        break
                if points is not None:
                    self.get_logger().info(f"Planned on: {name}")
                    break
        finally:
            self.clearance = map_clearance

        if points is None:
            return None
 
        # Benchmarking and logging the path to check its length and efficiency
        bench = self._plan(start, end, ROBOT_RADIUS, 0.0) # For minimal margin path (for comparing)
        length = path_length(points) # Length of the path
        min_clear = min(self.clearance[world_to_cell(x, y, self.origin, self.resolution)]
                        for x, y in points if self._inside(x, y)) # Minimum clearance in the path

        text = f"{len(points)} poses, {length:.2f} m, closest rock {min_clear:.2f} m (circle radius {ROBOT_RADIUS:.2f} m)"
        if bench is not None:
            text += f", benchmark estimate {path_length(bench):.2f} m, efficiency about {min(1.0, path_length(bench) / length):.2f}"
        self.get_logger().info("Planned: " + text)
 

        return points

    def _inside(self, x, y):
        '''
        Description:    Checks if a position is inside the grid.

        Args:
            x (float):    x-coordinate of the position
            y (float):    y-coordinate of the position

        Returns:
            bool:         True if the position is inside the grid, False otherwise
        '''
        row, col = world_to_cell(x, y, self.origin, self.resolution) # converting world coordinates to cell coordinates

        # Checking if the cell is within the grid boundaries and returning the boolean value
        return 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]
 
 
    def _plan(self, start, end, hard_radius, soft_band):
        '''
        Description:    One planning attempt at a given clearance.
 
        Args:
            start (tuple):       (x, y, yaw)
            end (tuple):         (x, y, yaw)
            hard_radius (float): cells closer than this to a rock are forbidden
            soft_band (float):   beyond hard_radius, cells this much further out cost extra

        Returns:
            list:                [(x, y), ...] or None
        '''

        # if start or end is outside the map, return None
        if not (self._inside(start[0], start[1]) and self._inside(end[0], end[1])):
            self.get_logger().error("A step's start or end is outside the map")
            return None
 
        # if hard_radius is less than or equal to 0, set blocked to False
        blocked = self.clearance <= hard_radius

        # if soft_band is greater than 0, calculate penalty
        if soft_band > 0.0:
            penalty = SOFT_WEIGHT * np.clip((hard_radius + soft_band - self.clearance) / soft_band, 0.0, 1.0)
        else:
            penalty = np.zeros(self.grid.shape)

        # converting start and end points to cell coordinates
        start_cell = world_to_cell(start[0], start[1], self.origin, self.resolution) # converting world coordinates to cell coordinates

        # converting end point to cell coordinates
        goal_cell = world_to_cell(end[0], end[1], self.origin, self.resolution) # converting world coordinates to cell coordinates
 
        # keep a small disc round each end free, but never a rock itself
        rows, cols = np.ogrid[:self.grid.shape[0], :self.grid.shape[1]]
        radius_cells = ENDPOINT_CLEAR_RADIUS / self.resolution

        # make the start and end points free
        for r0, c0 in (start_cell, goal_cell):
            disc = (rows - r0) ** 2 + (cols - c0) ** 2 <= radius_cells ** 2
            blocked[disc & (self.grid < OCCUPIED_THRESHOLD)] = False
 
        cells = astar(blocked, penalty, start_cell, goal_cell) # running A* algorithm
        if cells is None:
            return None
 
        cells = shorten_path(cells, blocked) # shortening the path
        points = [cell_to_world(r, c, self.origin, self.resolution) for r, c in cells] # converting cells to points
 
        # the cell centres are up to half a cell off; the path starts and ends exactly on the poses
        points[0] = (start[0], start[1])
        points[-1] = (end[0], end[1])
        return resample_path(points, RESAMPLE_SPACING)
 
 
    def publish_path(self, points, start, end):
        '''
        Description:    Publishes 'points' on /ebot_path as a nav_msgs/Path.

        Args:
            points (list):    [(x, y), ...] as returned by plan_path()
            start (tuple):    (x, y, yaw) where the step begins
            end (tuple):      (x, y, yaw) where the step ends

        Returns:
        '''

        path = Path() # creating a path
        path.header.frame_id = map_frame # setting the frame id of the path
        path.header.stamp = self.get_clock().now().to_msg() # setting the timestamp of the path

        last = len(points) - 1 # finding the last point in the path


        for i, (x, y) in enumerate(points): # iterating through the points in the path
            if i == 0: # if the point is the first point
                yaw = start[2] # set the yaw to the start yaw
            elif i == last: # if the point is the last point
                yaw = end[2] # set the yaw to the end yaw
            else:
                yaw = math.atan2(points[i + 1][1] - y, points[i + 1][0] - x) # calculating the yaw
 
            pose = PoseStamped() # creating a pose
            pose.header.frame_id = map_frame # setting the frame id of the pose
            pose.header.stamp = path.header.stamp # setting the timestamp of the pose
            
            # setting the position of the pose
            pose.pose.position.x = float(x) # setting the x-coordinate of the pose
            pose.pose.position.y = float(y) # setting the y-coordinate of the pose
            
            # setting the orientation of the pose
            pose.pose.orientation.z = math.sin(yaw / 2.0) # setting the z-coordinate of the pose
            pose.pose.orientation.w = math.cos(yaw / 2.0) # setting the w-coordinate of the pose
            path.poses.append(pose) # appending the pose to the path

        # publishing the path
        self.path_pub.publish(path)
 

    def request_ore_package(self):
        '''
        Description:    Calls /spawn_ore_package once the eBot stands still on the ore drop pose.

        Args:
        Returns:
        '''

        # checking if the spawn client is ready
        if not self.spawn_client.service_is_ready():
            self.get_logger().warn("/spawn_ore_package is not ready yet", throttle_duration_sec=2.0)
            return

        # setting the package pending flag to True
        self.package_pending = True

        # calling the spawn client
        future = self.spawn_client.call_async(Trigger.Request()) # calling the spawn client
        # adding a callback to the future
        future.add_done_callback(self.ore_package_done) # adding a callback to the future

    def ore_package_done(self, future):
        '''
        Description:    Runs when /spawn_ore_package replies.
 
        Args:
            future (Future):    the finished service call
 
        Returns:
        '''

        # setting the package pending flag to False
        self.package_pending = False

        # trying to get the result of the future
        try:
            reply = future.result()
        except Exception as error:
            self.get_logger().error(f"/spawn_ore_package failed: {error}")
            return
 
        # setting the package ok flag to True
        self.package_ok = bool(reply.success)
        # logging the reply
        self.get_logger().info(f"/spawn_ore_package replied success={reply.success}: '{reply.message}'")


    def advance_step(self):
        '''
        Description:    Move on to the next step of the lap, or finish after the last.
 
        Args:
        Returns:
        '''
        self.step += 1 # incrementing the step
        if self.step >= len(steps): # if the step is greater than the number of steps
            self.phase = 'DONE' # phase to done
            self.get_logger().info("Lap complete.") # logging that the lap is complete
        else:
            self.phase = 'PLAN' # phase to plan


    def process_planning(self):
        '''
        Description:    Timer function used to plan, publish and request, step by step.

        Args:
        Returns:
        '''

        # returning early if the map or odometry is not available
        if self.grid is None or self.odom is None:
            return 
 
        try:
            # getting the x, y, yaw, linear and angular from the odometry
            x, y, yaw, linear, angular = self.odom
            # getting the current time
            now = self.now_s()
 
            # checking if the eBot has stood still for a certain amount of time
            if abs(linear) > STILL_LINEAR or abs(angular) > STILL_ANGULAR:
                self.still_since = None
            elif self.still_since is None:
                self.still_since = now
            still_for = 0.0 if self.still_since is None else now - self.still_since

            # checking if the lap is complete
            if self.phase == 'DONE':
                return
 
            # getting the start and end points of the current step
            start, end = steps[self.step]

            # planning the path for the current step if the eBot is stationary
            if self.phase == 'PLAN':
                if still_for < PLAN_STILL_S:
                    return
                # planning the path
                points = self.plan_path(start, end)
                # checking if the path is not empty
                if points is None:
                    self.get_logger().error(f"No path for step {self.step + 1}", throttle_duration_sec=2.0)
                    return
                # publishing the path
                self.publish_path(points, start, end)
                # changing the phase to drive
                self.phase = 'DRIVE'
                # setting the phase time to now
                self.phase_t0 = now
                # logging that the path is published
                self.get_logger().info(f"Step {self.step + 1}: path published")

            # checking if the eBot has arrived at the end of the current step
            elif self.phase == 'DRIVE':
                at_end = (math.hypot(end[0] - x, end[1] - y) <= ARRIVE_POS_TOL
                          and abs(math.atan2(math.sin(end[2] - yaw), math.cos(end[2] - yaw))) <= ARRIVE_YAW_TOL
                          and still_for >= ARRIVE_STILL_S)
                if at_end:
                    self.get_logger().info(f"Step {self.step + 1}: arrived and standing still")
                    # if the step is 0, change the phase to package
                    if self.step == 0:
                        self.phase = 'PACKAGE'
                        self.package_tries = 0
                        self.last_package_try = -math.inf
                    # at the arm with the package on board: wait for the arm to take it
                    elif self.step == 1 and self.package_ok:
                        self.phase = 'HANDOVER'
                        self.phase_t0 = now
                        self.publish_at_arm(True)
                        self.get_logger().info("At the arm; waiting for the package to be taken")
                    else:
                        self.advance_step()
                elif now - self.phase_t0 > STEP_TIMEOUT_S:
                    self.get_logger().error(f"Step {self.step + 1} timed out")
                    self.advance_step()
            
            # checking if the package is pending
            elif self.phase == 'PACKAGE':
                if self.package_pending:
                    return
                if self.package_ok:
                    self.phase = 'SETTLE'
                    self.phase_t0 = now
                elif now - self.last_package_try >= PACKAGE_RETRY_S:
                    if self.package_tries >= MAX_PACKAGE_TRIES:
                        # missing the package costs only its own marks, so carry on with the lap
                        self.get_logger().error("Ore package never spawned; carrying on")
                        self.advance_step()
                        return
                    self.package_tries += 1
                    self.last_package_try = now
                    self.request_ore_package()
            # checking if the package is settled
            elif self.phase == 'SETTLE':
                if now - self.phase_t0 >= PACKAGE_SETTLE_S:
                    self.advance_step()
            # waiting beside the arm until the package is off the deck
            elif self.phase == 'HANDOVER':
                timed_out = now - self.phase_t0 > HANDOVER_TIMEOUT_S
                if timed_out:
                    self.get_logger().error("Package never taken; driving home anyway")
                if self.package_picked or timed_out:
                    self.publish_at_arm(False)
                    self.advance_step()
            # exception handling
        except Exception as error:
            self.get_logger().error(f"process_planning failed: {error}")


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_planner class to perform its task
    '''

    rclpy.init(args=sys.argv)                                           # initialisation

    node = rclpy.create_node('ebot_planner_process')                    # creating ROS node

    node.get_logger().info('Node created: eBot planner process')        # logging information

    ebot_planner_class = ebot_planner()                                 # creating a new object for class 'ebot_planner'

    try:
        rclpy.spin(ebot_planner_class)                                  # spining on the object to make it alive in ROS 2 DDS
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass                                                            # Ctrl + C between the two runs ends the node quietly
    finally:
        ebot_planner_class.destroy_node()                               # destroy node after spin ends
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()                                            # shutdown process


if __name__ == '__main__':

    main()
