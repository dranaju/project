#!/usr/bin/env python
#importing the packages
import rospy
from std_msgs.msg import Float32
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import urllib

#-------------------------------------------------------------------

rospy.init_node('vai')
pub_distance = rospy.Publisher('distance', Float32, queue_size=5)
distance32 = Float32()
pub_angle = rospy.Publisher('angle', Float32, queue_size=5)
angle32 = Float32()

# ----------------------------------------------------------------

def mid_point(a,b):
	return (int((a[0]+b[0])/2),int((a[1]+b[1])/2))
def vector(a,b):
	return((a[0]-b[0]),(a[1]-b[1]))
def vet_sum(a,b):
	return((a[0]+b[0]),(a[1]+b[1]))
def calculate_angle(a,b):
	x1 = a[0]
	y1 = a[1]

	x2 = b[0]
	y2 = b[1]

	return np.arccos((x1*x2+y1*y2)/( ((x1**2+y1**2)**0.5) * ((x2**2+y2**2)**0.5) ))

def perpendicular(a):
	return np.array([-a[1],a[0]])

def modulo(a):

	return ((a[0]**2+a[1]**2)**0.5) 


# Function returns N largest elements 
def Nmaxelements(list1, N): 
	final_list = [] 
	if len(list1) > N:
		for i in range(0, N):  
			max1 = 0
			cmax = 0
			index = 0
			for i,c in enumerate(list1):      
				if cv2.contourArea(c) > max1: 
					max1 = cv2.contourArea(c)
					cmax = c
					index = i
			final_list.append(cmax) 
			list1.pop(index) 
	else:
		return list1
	return (final_list) 

#cap = cv2.VideoCapture('20190403_143347.mp4')
url = 'http://192.168.1.106:8080/shot.jpg'

blueLower = (58,108,199)
blueUpper = (136,255,255)

greenLower = (26,120,109)
greenUpper = (128,255,203)

redLower =(141,90,90)
redUpper =(220,255,255)

yellowLower =(0,88,153)
yellowUpper =(65,255,255)

#getting frames
while (True):
	imgResp = urllib.urlopen(url)
	imgNp = np.array(bytearray(imgResp.read()), dtype = np.uint8)
	frame = cv2.imdecode(imgNp, -1)
	#ret, frame = cap.read()


	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "blue", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, blueLower, blueUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	leftPoint = None
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid

		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		leftPoint = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

	mask = cv2.inRange(hsv, redLower, redUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	rightPoint = None
	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		rightPoint = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		
	if rightPoint and leftPoint:	
		midPoint = mid_point(rightPoint, leftPoint)

	#yellow aimPoint
	mask = cv2.inRange(hsv, yellowLower, yellowUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	aimPoint = None
	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		aimPoint = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
	
	#greenPoints 
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	green1 = None
	green2 = None

	if len(cnts) > 1:

		cnts = Nmaxelements(cnts, 2) 
		#green1	cnts[0]
		((x, y), radius) = cv2.minEnclosingCircle(cnts[0])
		M = cv2.moments(cnts[0])
		green1 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		

		#green2 cnts[1]
		((x, y), radius) = cv2.minEnclosingCircle(cnts[1])
		M = cv2.moments(cnts[1])
		green2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
	
	if aimPoint and rightPoint and leftPoint and green1:
		vectorTurtle = np.array(vector(leftPoint,rightPoint))
		vectorDistance = np.array(vector(aimPoint,midPoint))
		cv2.circle(frame, leftPoint, 5, (0, 0, 204), -1)
		cv2.circle(frame, rightPoint, 5, (204, 0, 0), -1)
		cv2.circle(frame, aimPoint, 5, (0, 150, 254), -1)
		cv2.circle(frame, green1, 5, (0, 255, 0), -1)
		cv2.circle(frame, green2, 5, (0, 255, 0), -1)
		cv2.circle(frame, midPoint, 5, (200, 200, 200), -1)
		cv2.line(frame, rightPoint, vet_sum(rightPoint,vectorTurtle), (255,255,255), thickness=1, lineType=8, shift=0) 
		cv2.line(frame, midPoint, vet_sum(midPoint,vectorDistance), (255,255,255), thickness=1, lineType=8, shift=0) 


		vectorPerp = perpendicular(vectorTurtle)

		print("vectorPerp", vectorPerp)
		cv2.line(frame, midPoint, vet_sum(midPoint,vectorPerp), (255,255,255), thickness=1, lineType=8, shift=0) 

		angle1 = calculate_angle(vectorPerp, vectorDistance)
		angle2 = calculate_angle(vectorTurtle, vectorDistance)


		
		if angle2 > np.pi/2:
			angle1*=(-1)
		print(angle1)

		moduloDistance = modulo(vectorDistance)
		moduloTurtle = modulo(vectorTurtle)
		vectorGreen = np.array(vector(green1,green2))
		moduloGreen = modulo(vectorGreen)

		greenDistance = (moduloGreen*0.171)/moduloTurtle
		turtleDistance = (moduloTurtle*1.5)/moduloGreen
		distance = (moduloDistance*1.5)/moduloGreen
		print('distancia em m',distance)

		#-----------------------------------------------------
		angle32 = angle1
		distance32 = distance
		pub_distance.publish(distance32)
		pub_angle.publish(angle32)



		cv2.line(frame, green1, green2, (0,250,0), thickness=1, lineType=8, shift=0) 

		cv2.imshow('mask',mask)

		cv2.imshow('frame',frame)
		
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()