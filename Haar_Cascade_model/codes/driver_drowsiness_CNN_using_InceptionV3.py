#!/usr/bin/env python
# coding: utf-8
#Author:- Basu Verma (142002007)


#Importing all required modules
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
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import cv2


#Reading closed and open eyes and converting them to grayscale image
import os
classes=['closed_eyes','open_eyes']
def create_training_Data():
    for category in classes:
        path = os.path.join('G:\Mtec_Final_year_project_research paper\datasets\mrlEyes_2018_01 - Copy - Copy',category)
        class_num=classes.index(category)
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
            new_array = cv2.resize(backtorgb,(150,150))
            training_data.append(new_array/255.0)
            label.append(class_num)

training_data=[]
label=[]
create_training_Data()


#Shuffling the data 
from sklearn.utils import shuffle
X, y = shuffle(training_data, label, random_state=0)


#Converting data into numpy array
X = np.array(X)
y= np.array(y)



#Downloading Inception model wiht input size as 150x150
basemodel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(150, 150, 3)))
print(basemodel.summary())
headModel = basemodel.output   #Output layer of basemodel
headModel = Flatten(name="flatten")(headModel)  # Flatten layer
headModel = Dense(64, activation="relu")(headModel)  # First hidden layer
headModel = Dense(1, activation="sigmoid")(headModel)  # Output layer

# Making layers of basemodel as untrainable layers 
for layer in basemodel.layers:
    layer.trainable = False

model = Model(inputs=basemodel.input, outputs=headModel) #Final model
model.summary()

#Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#Training the model with validation split of 20%
H = model.fit(X,y,epochs=40,validation_split=0.2)

#Saving the model
model.save('drowsiness_model_inception_1.h5')


#Plotting Accuracy and loss graphs
print("[INFO] Training curve with loss and accuracy********")
#  "Training Accuracy"
plt.plot(H.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# "Validation Accuracy"
plt.plot(H.history['val_accuracy'])
plt.title('model validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['validation'], loc='upper left')
plt.show()


# Some Prediction


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

