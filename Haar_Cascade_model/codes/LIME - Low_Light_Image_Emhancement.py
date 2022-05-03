#!/usr/bin/env python
# coding: utf-8
#Author = Basu Verma
#Link:- https://in.mathworks.com/help/images/low-light-image-enhancement.html
'''Paper link:- 
X. Guo, Y. Li and H. Ling, "LIME: Low-Light Image Enhancement via Illumination Map Estimation," in IEEE Transactions on Image Processing, vol. 26, no. 2, pp. 982-993, Feb. 2017, doi: 10.1109/TIP.2016.2639450.'''

#Importing all required packages
import matplotlib.pyplot as plt
import cv2
import numpy as np
import image_dehazer


#load an image
img = cv2.imread("2437.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
print('Shape: ',img.shape)


#Max and Min intensity
max_intensity_R = np.max(img[:,:,0])
max_intensity_G = np.max(img[:,:,1])
max_intensity_B = np.max(img[:,:,2])
min_intensity_R = np.min(img[:,:,0])
min_intensity_G = np.min(img[:,:,1])
min_intensity_B = np.min(img[:,:,2])
print('Max Intensity R: ',max_intensity_R)
print('Min Intensity R: ',min_intensity_R)
print('Max Intensity G: ',max_intensity_G)
print('Min Intensity G: ',min_intensity_G)
print('Max Intensity B: ',max_intensity_B)
print('Min Intensity B: ',min_intensity_B)


#Invert the image
new_img = 255-img
plt.imshow(new_img)


#Dehazing the inverted image
dehaze_img = image_dehazer.remove_haze(new_img)
plt.imshow(dehaze_img)


#inverse the dehaze image
inv_img = 255-dehaze_img
plt.imshow(inv_img)


#Further reduce haze
#Dehazing the new image
final_img = image_dehazer.remove_haze(inv_img)
plt.imshow(final_img)

#Show final output after LIME
output = np.hstack((img,final_img))
plt.figure(figsize = (20,40))
plt.imshow(output)


#Write the imager
img_write = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
cv2.imwrite('imwrite_eg.jpg',img_write) #write as RGB





