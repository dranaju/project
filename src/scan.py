#! /usr/bin/env python
import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

rospy.init_node('test')

while not rospy.is_shutdown():
    scan_range = []
    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('scan', LaserScan, timeout= 5)
        except:
            pass
    scan = data.ranges
    scan_range.append(max(scan[88], scan[89], scan[90]))
    scan_range.append(max(scan[68], scan[69], scan[70]))
    scan_range.append(max(scan[48], scan[49], scan[50]))
    scan_range.append(max(scan[28], scan[29], scan[30]))
    scan_range.append(max(scan[8], scan[9], scan[10]))
    scan_range.append(max(scan[-10], scan[-11], scan[-12]))
    scan_range.append(max(scan[-30], scan[-31], scan[-32]))
    scan_range.append(max(scan[-50], scan[-51], scan[-52]))
    scan_range.append(max(scan[-70], scan[-71], scan[-72]))
    scan_range.append(max(scan[-90], scan[-91], scan[-92]))
    for i in range(len(scan_range)):
        scan_range[i] = round(scan_range[i], 3)
    print('vai:', scan_range)