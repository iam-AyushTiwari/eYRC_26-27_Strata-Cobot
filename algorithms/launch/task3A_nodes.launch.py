#!/usr/bin/env python3
'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This launch file starts the four Task 3A nodes of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          1454
# Author List:      Aryan Joshi, Ayush Tiwari
# Filename:         task3A_nodes.launch.py
# Functions:        generate_launch_description
# Nodes:            Starts task3A_perception.py, task3A_manipulation.py,
#                   task3A_planning.py and task3A_navigation.py
#
# Start it only once the evaluator prints Ready (or the world prints
# "Task 3A world ready"). It does NOT start the world:
#
#     ros2 launch algorithms task3A_nodes.launch.py


################### IMPORT MODULES #######################

from launch import LaunchDescription
from launch_ros.actions import Node


##################### TASK CONSTANTS #######################

task3a_scripts = [                          # the four Task 3A nodes, by executable name
    'task3A_perception.py',
    'task3A_manipulation.py',
    'task3A_planning.py',
    'task3A_navigation.py',
]


##################### FUNCTION DEFINITION #######################

def generate_launch_description():
    '''
    Description:    Builds the launch description that starts every Task 3A node.

    Args:
    Returns:
        launch_description (LaunchDescription):    the four nodes, each printing to the screen
    '''
    return LaunchDescription([
        Node(package='algorithms', executable=script, output='screen', emulate_tty=True)
        for script in task3a_scripts
    ])
