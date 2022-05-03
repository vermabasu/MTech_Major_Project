#!/usr/bin/env python
# coding: utf-8
#Author:- Basu Verma

#Import required packages
import cv2
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


#Read all the closed and open eyes images 
import os
classes=['closed_eyes','open_eyes']
def create_training_Data():
    for category in classes:
        path = os.path.join('G:\Mtec_Final_year_project_research paper\datasets\mrlEyes_2018_01 - Copy - Copy',category)
        class_num=classes.index(category)
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # Reading images in grayscale.
            backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)     # changing the channel of image from 1 to 3.
            new_array = cv2.resize(backtorgb,(150,150))            #Resizing image to 150x150.
            training_data.append(new_array/255.0)               #Normalizing each image between 0 and 1.
            label.append(class_num)                         #Appending class of each image.

training_data=[]
label=[]
create_training_Data()


#Shuffle all the images available 
from sklearn.utils import shuffle
X, y = shuffle(training_data, label, random_state=0)


#Converting all images and corresponding labels to numpy array
X = np.array(X)
y= np.array(y)




#Downloading Pretrained model VGG16
basemodel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(150, 150, 3)))
print(basemodel.summary())
headModel = basemodel.output  #Output of basemodel
headModel = Flatten(name="flatten")(headModel)  # Flatten layer
headModel = Dense(64, activation="relu")(headModel)  # First hidden layer
headModel = Dense(1, activation="sigmoid")(headModel)  # Output layer

# Making pretrained layers training as False
for layer in basemodel.layers:
    layer.trainable = False

model = Model(inputs=basemodel.input, outputs=headModel) #Final Model
model.summary()


#Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Uncomment below for early stopping of training to reduce time
# class myCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self,epoch,logs={}):
#         if logs.get('accuracy')>0.9:
#             print("\nAccuracy is achieved. So cancelling training!")
#             self.model.stop_training=True
# newcallbacks=myCallback


#Uncomment below for using inbuilt earltstopping function
# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping()

  #Training the model
H = model.fit(X,y,epochs=40,validation_split=0.2) 

#Saving the model
model.save('drowsiness_model_vgg_best.h5')


#Printing the training accuracy and validation accuracy of above model
print("[INFO] Training curve with loss and accuracy********")
#  "Accuracy"
plt.plot(H.history['accuracy'])
plt.title('model Training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# "Loss"
plt.plot(H.history['val_accuracy'])
plt.title('model validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['validation'], loc='upper left')
plt.show()


# # Some Prediction
input_img = cv2.imread('s0023_00025_0_0_0_0_1_01.png', cv2.IMREAD_GRAYSCALE)
input_image_backtorgb = cv2.cvtColor(input_img,cv2.COLOR_GRAY2RGB)
new_input_array = cv2.resize(input_image_backtorgb,(150,150))
new_input_array = np.array(new_input_array).reshape(-1,150,150,3)
new_image = new_input_array/255.0
plt.imshow(input_image_backtorgb)
prediction = model.predict(new_image)
prediction




input_img = cv2.imread('s0023_00300_0_0_1_0_1_01.png', cv2.IMREAD_GRAYSCALE)
input_image_backtorgb = cv2.cvtColor(input_img,cv2.COLOR_GRAY2RGB)
new_input_array = cv2.resize(input_image_backtorgb,(150,150))
new_input_array = np.array(new_input_array).reshape(-1,150,150,3)
new_image = new_input_array/255.0
plt.imshow(input_image_backtorgb)
prediction = model.predict(new_image)
prediction



input_img = cv2.imread('s0028_00007_0_0_0_0_0_01.png', cv2.IMREAD_GRAYSCALE)
input_image_backtorgb = cv2.cvtColor(input_img,cv2.COLOR_GRAY2RGB)
new_input_array = cv2.resize(input_image_backtorgb,(150,150))
new_input_array = np.array(new_input_array).reshape(-1,150,150,3)
new_image = new_input_array/255.0
plt.imshow(input_image_backtorgb)
prediction = model.predict(new_image)
prediction




