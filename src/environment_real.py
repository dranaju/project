#! /usr/bin/env python
import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

class Env():
    def __init__(self):
        self.heading = 0
        #self.initGoal = True
        self.get_goalbox = False
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.sub_angle = rospy.Subscriber('angle', Float32, self.getAngle)
        self.sub_distance = rospy.Subscriber('distance', Float32, self.getDistance)
        self.angle = 0.
        self.distance = 0.
        self.past_distance = 0.
        #Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        #you can stop turtlebot by publishing an empty Twist
        #message
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)

    def getAngle(self, msg):
        self.angle = msg.data

    def getDistance(self, msg):
        self.distance = msg.data

    def getGoalDistace(self):
        pass

    def getOdometry(self, odom):
        pass

    def getState(self, scan, past_action):
        scan_range = []
        heading = round(self.angle, 2)
        min_range = 0.12  
        done = False

        scan_range.append(max(scan[-90], scan[-91], scan[-92]))
        scan_range.append(max(scan[-70], scan[-71], scan[-72]))
        scan_range.append(max(scan[-50], scan[-51], scan[-52]))
        scan_range.append(max(scan[-30], scan[-31], scan[-32]))
        scan_range.append(max(scan[-10], scan[-11], scan[-12]))
        scan_range.append(max(scan[8], scan[9], scan[10]))
        scan_range.append(max(scan[28], scan[29], scan[30]))
        scan_range.append(max(scan[48], scan[49], scan[50]))
        scan_range.append(max(scan[68], scan[69], scan[70]))
        scan_range.append(max(scan[88], scan[89], scan[90]))

        for i in range(len(scan_range)):
            if scan_range[i] == 0.0:
                scan_range[i] = 3.5
            elif scan_range[i] > 3.5:
                scan_range[i] = 3.5
            scan_range[i] = round(scan_range[i], 3)
        #print('vai:', scan_range)

        for j in range(len(scan_range)):
            if scan_range[j] == float('Inf'):
                scan_range[j] = 3.5
            elif np.isnan(scan_range[j]):
                scan_range = 0.


        if min_range > min(scan_range) > 0:
            done = True

        for pa in past_action:
            scan_range.append(pa)

        current_distance = round(self.distance,2)
        if current_distance < 0.2:
            self.get_goalbox = True

        return scan_range + [heading, current_distance], done

    def setReward(self, state, done):
        current_distance = state[-1]
        heading = state[-2]
        #print('cur:', current_distance, self.past_distance)


        distance_rate = (self.past_distance - current_distance) 
        if distance_rate > 0:
            reward = 200.*distance_rate
        #if distance_rate == 0:
        #    reward = -10.
        if distance_rate <= 0:
            reward = -8.
        #angle_reward = math.pi - abs(heading)
        #print('d', 500*distance_rate)
        #reward = 500.*distance_rate #+ 3.*angle_reward
        self.past_distance = current_distance

        if done:
            rospy.loginfo("Collision!!")
            reward = -550.
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 500.
            self.pub_cmd_vel.publish(Twist())
            #self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            #self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data.ranges, past_action)
        reward = self.setReward(state, done)

        return np.asarray(state), reward, done

    def reset(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data.ranges, [0.,0.])

        return np.asarray(state)
