#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:17:57 2020

@author: yash
"""

import cv2

# Creating a cascaded classifier object
face_cascade = cv2.CascadeClassifier("/home/yash/Basic/Computer Science/Data Science/Projects/Face Detection/haarcascade_frontalface_alt.xml")

# Reading the image
img = cv2.imread("/home/yash/Basic/Computer Science/Data Science/Projects/Face Detection/face_photo.jpg", 1)

# Reading the image as gray scale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Searching the coordinates of image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)

print(type(faces))
print(faces)

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
resized = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))

cv2.imshow("Gray", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()