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

dados_xy = 'dados_teste_xy.csv'
alvo_xy = 'alvo_teste_xy.csv'
ref_pixel_meter = 'pixel_meter_teste_.csv'

filename = open(dados_xy,'w')
filename.close()

filename1 = open(alvo_xy,'w')
filename1.close()

filename2 = open(ref_pixel_meter,'w')
filename2.close()

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
def vet_sum_dif(a,b):
	return(abs((a[0]-b[0]))+abs((a[1]-b[1])))
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

#calculate the 10 distance angle vector
def lidar_dist(vector):
	unit_vectors = [complex(1,0),complex(0.939,0.342),complex(0.766,0.642),complex(0.599,0.866), \
					complex(0.173,0.984),complex(-0.173,0.984),complex(-0.599,0.866), \
					complex(-0.766,0.642),complex(-0.939,0.342),complex(-1,0)]	
	output = []
	for u,v in zip(unit_vectors, vector):
		pass


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

#calculate the 10 distance angle vector
def lidar_dist(vector,distances,conversion):
	#calculando o vetor unitario
	d = (vector[0]**2+vector[1]**2)**0.5 
	vector = [vector[0]/d,vector[1]/d]
	vector = complex(vector[0],vector[1])

	unit_vectors = [complex(1,0),complex(0.939,0.342),complex(0.766,0.642),complex(0.599,0.866), \
					complex(0.173,0.984),complex(-0.173,0.984),complex(-0.599,0.866), \
					complex(-0.766,0.642),complex(-0.939,0.342),complex(-1,0)]	
	output = []
	for u,d in zip(unit_vectors, distances):
		u = u * vector
		a = u * complex(d,0)
		b = [int(a.real*conversion),int(a.imag*conversion)]
		output.append(b)

	return output

def get_angle_distance(frame,greenDistance,pts,lidar):
	# resize the frame, blur it, and convert it to the HSV
	# color space

	blueLower = (58,108,199)
	blueUpper = (136,255,255)

	greenLower = (26,120,109)
	greenUpper = (128,255,203)

	redLower =(141,90,90)
	redUpper =(220,255,255)

	yellowLower =(0,115,153)
	yellowUpper =(65,255,255)
	#frame = imutils.resize(frame, )
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
	green1 = 0
	green2 = 0
	if len(cnts)>1:
		cnts = Nmaxelements(cnts, 2)

		#green1	cnts[0]
		((x, y), radius) = cv2.minEnclosingCircle(cnts[0])
		M = cv2.moments(cnts[0])
		green1 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		

		#green2 cnts[1]
		((x, y), radius) = cv2.minEnclosingCircle(cnts[1])
		M = cv2.moments(cnts[1])
		green2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
	
	if leftPoint and rightPoint and aimPoint and midPoint and green1 and green2:


		
		vectorTurtle = np.array(vector(leftPoint,rightPoint))
		vectorDistance = np.array(vector(aimPoint,midPoint))
		
		#drawing the robot tracking in blue
		for i in range(1, len(pts)):
			# if either of the tracked points are None, ignore
			# them
			if pts[i - 1] is None or pts[i] is None:
				continue
	 
			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			cv2.line(frame, pts[i - 1], pts[i], (255, 0, 0), 4)
 
		#desenhando os pontos e as linhas
		cv2.circle(frame, leftPoint, 5, (0, 0, 204), -1)
		cv2.circle(frame, rightPoint, 5, (204, 0, 0), -1)
		cv2.circle(frame, aimPoint, 5, (0, 150, 254), -1)
		cv2.circle(frame, green1, 5, (0, 255, 0), -1)
		cv2.circle(frame, green2, 5, (0, 255, 0), -1)
		cv2.circle(frame, midPoint, 5, (200, 200, 200), -1)
		cv2.line(frame, rightPoint, vet_sum(rightPoint,vectorTurtle), (255,255,255), thickness=1, lineType=8, shift=0) 
		cv2.line(frame, midPoint, vet_sum(midPoint,vectorDistance), (255,255,255), thickness=1, lineType=8, shift=0) 
		#print(vectorTurtle)
		#print(vectorDistance)

		vectorPerp = perpendicular(vectorTurtle)


		cv2.line(frame, midPoint, vet_sum(midPoint,vectorPerp), (255,255,255), thickness=1, lineType=8, shift=0) 

		angle1 = calculate_angle(vectorPerp, vectorDistance)
		angle2 = calculate_angle(vectorTurtle, vectorDistance)


	
		if angle2 > np.pi/2:
			angle1*=(-1)

		moduloDistance = modulo(vectorDistance)
		moduloTurtle = modulo(vectorTurtle)
		vectorGreen = np.array(vector(green1,green2))
		moduloGreen = modulo(vectorGreen)

		pixel_metro = moduloGreen/greenDistance

		distance = (moduloDistance*greenDistance)/moduloGreen

		if pts == []:
			pts.append(midPoint)
		#points distance
		elif vet_sum_dif(pts[-1],midPoint) > 10:
			pts.append(midPoint)
			print('pontoxy: ' + str(midPoint))
			print('pixel metro', pixel_metro)
			with open(ref_pixel_meter, 'a') as f:
				f.write(str(pixel_metro) + '\n')
			with open(dados_xy, 'a') as f:
				f.write(str(midPoint[0]) + ',' + str(midPoint[1]) + '\n')
			with open(alvo_xy, 'a') as f:
				f.write(str(aimPoint[0]) + ',' + str(aimPoint[1]) + '\n')
			

		#10 vetores distancia
		conversion = moduloGreen/greenDistance
		vectors = lidar_dist(vectorTurtle,lidar,conversion)
		#print('v')
		for v in vectors:

			cv2.line(frame, midPoint, vet_sum(midPoint,v), (255,0,5), thickness=1, lineType=8, shift=0) 
			#print(v)

		cv2.line(frame, green1, green2, (0,250,0), thickness=1, lineType=8, shift=0) 

		cv2.imshow('frame',frame)

		#time.sleep(0.01)
		return (angle1, distance)
	else:
		return (None,None)



def get_ang_dist(green,pts,lidar):
	
	
	imgResp = urllib.urlopen(url)
	imgNp = np.array(bytearray(imgResp.read()), dtype = np.uint8)
	frame = cv2.imdecode(imgNp, -1)
	#ret, frame = cap.read()

	#if (cap.isOpened()):
	#frame, distance of the green circles
	return get_angle_distance(frame, green,pts,lidar)

	#else:
	#	cap.release()
	#	cv2.destroyAllWindows()
	#	return (0,0)





#aqui q comeca
url = 'http://192.168.1.106:8080/shot.jpg'

from sensor_msgs.msg import LaserScan

#points to robot tracking 
pts = []



while(True):
    #print('asd_____________________')
    scan_range = []
    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('scan', LaserScan, timeout= 5)
        except:
            pass
    #print('asd_____________________')
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
        if scan_range[i] == 0.0:
            scan_range[i] = 3.5
        elif scan_range[i] > 3.5:
            scan_range[i] = 3.5
        scan_range[i] = round(scan_range[i], 3)
    #print('lidar: ', scan_range)
    lidar = scan_range
	#sending real green distance (here, 1,17 meters) 
    angle, distance = get_ang_dist(1.5,pts,lidar)

    #-----------------------------------------------------
    angle32 = angle
    distance32 = distance
    pub_distance.publish(distance32)
    pub_angle.publish(angle32)
    #---------------------------------------------------

    if distance == 0 or (cv2.waitKey(1) & 0xFF == ord('q')):
        break
    else:
        print('Angulo = %.4f    -    Distance = %.2f' % (angle,distance))


cv2.destroyAllWindows()
