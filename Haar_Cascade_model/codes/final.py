#!/usr/bin/env python
# coding: utf-8


#Importing all required packages and libraries
import cv2
import matplotlib.pyplot as plt
import image_dehazer
import numpy as np
from threading import Thread


#function to crop the image to increase the speed of face detection
def cropping(img):
    height = img.shape[0]
    width = img.shape[1]
    new_h = height//4
    new_w = width//4
    new_img = img[new_h:int(new_h*3),new_w:int(new_w*3)]
    return new_img


#function to implement image enhancement
'''LIME'''
def LIME(img):
    #Invert the image
    new_img = 255-img 
    #Dehazing the inverted image
    dehaze_img = image_dehazer.remove_haze(new_img)
    #inverse the dehaze image
    inv_img = 255-dehaze_img    
    return inv_img


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                

    def stop(self):
        self.stopped = True

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True


eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
def drowsy(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_classifier.detectMultiScale(gray,1.3,4)
    if len(eyes) == 0:
        print("eyes are not detected")
    else:
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
    return frame

def threadBoth():
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(0).start()
    video_shower = VideoShow(video_getter.frame).start()
    
    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        
        #cropping image
        frame = cropping(frame)
    
        #Low-light-image-enhancement
        inv_img = LIME(frame)

        #detect eye
        new_img = drowsy(inv_img)
        
        video_shower.frame = inv_img

threadBoth()

