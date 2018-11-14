# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 18:19:08 2017

@author: user
"""

import cv2
import matplotlib.pyplot as plt
 
# capture frames from a video
frames = cv2.imread('malo.jpg') 
# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cars.xml')
gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
cars = car_cascade.detectMultiScale(gray, 1.1, 1)
gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
cars = car_cascade.detectMultiScale(gray, 1.1, 1)
print len(cars)
for (x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)     

plt.imshow(cv2.cvtColor(frames, cv2.COLOR_BGR2RGB))
"""
while True:
    # reads frames from a video
    
    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
     
 
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
     
    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
 
   # Display frames in a window 
    cv2.imshow('', frames)
     
    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break
 
# De-allocate any associated memory usage
cv2.destroyAllWindows()
"""