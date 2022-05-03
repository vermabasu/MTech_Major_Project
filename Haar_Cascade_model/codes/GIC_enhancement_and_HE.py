#!/usr/bin/env python
# coding: utf-8
#Author:- Basu Verma

#Histogram plotting of gray-scale image
import matplotlib.pyplot as plt
import cv2
img = cv2.imread("WIN_20211028_15_58_50_Pro.jpg",1)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print('Shape of grayscale image: ',img.shape)
print(img)


#GIC
import numpy as np

#Function to adjust gamma correction
def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

cv2.imshow('original image',img)
gamma = 1.2                                # change the value here to get different result
adjusted = adjust_gamma(img_gray, gamma=gamma)
cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
cv2.imshow("gammam image 1", adjusted)

histr = cv2.calcHist([adjusted],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()



facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(adjusted,1.1,4)
for (x,y,w,h) in eyes:
    cv2.rectangle(adjusted,(x,y),(x+w,y+h),(0,255,0),2)
plt.imshow(adjusted,cmap='gray')




# # Histogram Equalization


dst = cv2.equalizeHist(img_gray)
hist_HE = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.plot(hist_HE)
plt.show()


cv2.imshow("Histogram Equalization enhancement of original image", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(dst,1.1,4)
for (x,y,w,h) in eyes:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


plt.figure(figsize=(20,50))
plt.imshow(img)


