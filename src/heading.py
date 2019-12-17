#!/usr/bin/env python
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


goal_x = 0
goal_y = 0
heading = 0
initGoal = True
respawn_goal = Respawn()

if initGoal:
    goal_x, goal_y = respawn_goal.getPosition()
    initGoal = False

def getOdometry(odom):
    position = odom.pose.pose.position
    orientation = odom.pose.pose.orientation
    orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
    _, _, yaw = euler_from_quaternion(orientation_list)

    goal_angle = math.atan2(goal_y - position.y, goal_x - position.x)

    print 'yaw', yaw
    print 'gA', goal_angle

    heading = goal_angle - yaw
    if heading > pi:
        heading -= 2 * pi

    elif heading < -pi:
        heading += 2 * pi
    
    print 'heading', heading
    heading = round(heading, 3)

rospy.init_node('test3')
sub_odom = rospy.Subscriber('odom', Odometry, getOdometry)
rospy.spin()