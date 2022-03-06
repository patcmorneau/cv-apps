
import sys
sys.path.insert(0, '../mathlib') # https://github.com/patcmorneau/mathlib
import mathlib

import os
import CVLIB
import cv2 as cv
import numpy as np
import numpy.linalg as LA
import dlib

def calculate_accuracy(val, val2):
	total = val + val2
	ans = total * 100/33
	return int(ans)

def smile_detection(image,landmarks, rect, showInfo=False):
	pt1 = landmarks[54] # right mouth corner
	pt2 = landmarks[60] # left
	landmarks = landmarks[50:68]
	height, width ,chan = image.shape
	center = mathlib.calculate_center(landmarks)
	horizontal_vector = mathlib.unit_vector(np.array([width - center[0], 0]))
	right_vector = mathlib.unit_vector(np.array([pt1[0]- center[0] ,center[1] - pt1[1]])) # since image height are postive towards bottom, we will substract heights the opposite way
	left_vector = mathlib.unit_vector(np.array([pt2[0] - center[0], center[1] - pt2[1]]))
	if(showInfo == True):
		print("hv: ", horizontal_vector)
		print("rv: ", right_vector)
		print("lv: ", left_vector)
		print("height", height)
		print("center, pt1, pt2 ", center, pt1, pt2)
		cv.circle(image, (center[0], center[1]), 1, (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
		cv.circle(image, pt1, 1, (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
		cv.circle(image, pt2, 1, (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
		cv.line(image, center, pt1, (255, 200, 0), thickness=2)
		cv.line(image, center, pt2, (255, 200, 0), thickness=2)
		cv.line(image, center , (width, center[1]), (255, 200, 0), thickness=2)
		
	if(right_vector[1] > 0 and left_vector[1] > 0):
		right_angle = mathlib.angle_between(horizontal_vector, right_vector)
		left_angle = mathlib.angle_between(horizontal_vector, left_vector)
		accuracy =  calculate_accuracy(right_angle, left_angle)
		label = "Smilling " + str(accuracy) + "%"
		cv.putText(image,label, (rect[1][0] - 50, rect[0][1] - 10 ), cv.FONT_HERSHEY_SIMPLEX, 1/3, (0,255,0),1, cv.LINE_AA)
		if(showInfo ==  True):
			print("R angle : ", right_angle)
			print("L angle : ", left_angle)
		# smilling %
	else:
		cv.putText(image,"Not smilling", (rect[1][0] - 50 , rect[0][1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1/3, (0,0,255),1, cv.LINE_AA)


first_frame = None
cap = cv.VideoCapture(0)
x = 0
landmarks = []
rect2 = []
print("testing")
while True:
	check, im = cap.read()
	
	scale_percent = 50 # percent of original size
	width = int(im.shape[1] * scale_percent / 100)
	height = int(im.shape[0] * scale_percent / 100)
	frame = cv.resize(im, (width, height), interpolation= cv.INTER_LINEAR)
	rect = CVLIB.detect_face(frame)
	if(len(rect) == 1):
		rect = dlib.rectangle(int(rect[0].left()),int(rect[0].top()),int(rect[0].right()),int(rect[0].bottom()))
		rect2 = [[rect.left(), rect.top()], [rect.right(), rect.bottom()]]
		#print(rect2[0][1])
		frame = cv.rectangle(frame, rect2[0], rect2[1], (255,0,0), 2)
	if x == 0 or x % 5 == 0:
		#print(rect2, type(rect2))
		landmarks = CVLIB.get_landmarks(frame, rect)
		
	if(len(landmarks) == 68):
		smile_detection(frame, landmarks, rect2, True)
	
	#frame = CVLIB.draw_landmarks(frame,landmarks)
	
	frame = cv.resize(frame, (width*2, height*2), interpolation= cv.INTER_LINEAR)
	cv.imshow('frame',frame)
	x = x+1
	if cv.waitKey(1) & 0xFF == ord('q'):
		break;

cap.release()
cv.destroyAllWindows()

