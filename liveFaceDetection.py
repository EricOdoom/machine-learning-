#importing libraries
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

#initializing face detection module
face_classifier = cv2.CascadeClassifier('Master OpenCV/Haarcascades/haarcascade_frontalface_default.xml')

#setting up function to detect faces
def face_detector(img, size=0.8):
	#converting live image fed to gray first for easy analysis by the face detection module
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#alowing the face detector to detect muktiple faces simultaneously
	faces = face_classifier.detectMultiScale(gray, 1.3,8)
	if faces is ():
		return img 

	for (x,y,w,h) in faces:
		
		#drawing a rectangle around detected face
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
		
		


		
	#changing live image back to RBG form from gray 
	img_col = cv2.flip(img, 1)
	return img_col
#initializing live video feed
cap =cv2.VideoCapture(0)
#Loop for continuous image detection
while True:
	ret, frame = cap.read()
	cv2.imshow('Face detection', face_detector(frame))
	if cv2.waitKey(1) == 13:
		break

