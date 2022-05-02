# MTech_Major_Project

Topic:- Driver Drowsiness Detection

There are several methods to detect a driver’s drowsiness. The one discussed here is driver drowsiness based on eye detection. For this, two methods are discussed here.
     * The first one is based on Haar Cascade features. In this, real time video is fed into the proposed algorithm. The algorithm first enhances each video frame using the technique ”LIME”. Then the face is detected and the eye portion is taken from it. The reason behind this is to reduce the complexity and time for eye detection. After that, for eye detection, the Haar Cascade Classifier is used here. After eye detection, the CNN model is used for classifying it into an open or close state and blink frequency/PERCLOS is calculated from it. It can tell whether a driver is in a drowsy state or not. 
     * The second one is based on the YoLov5 model. In this, detection of the eye and classification of the same into open eye and closed eye is happening under the same block. After that, blink frequency/PERCLOS is calculated to make the decision about a drowsy state.
<img src="inventory/results.jpg">
