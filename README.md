# SelfDrivingCar-BehavioralCloning

This is the third project in the Self Driving Car Nanodegree course offered by Udacity.

In this project a car is taught to run autonomously by cloning the behavior of manually driven car. First lot of good driving data (camera images of the road and steering angle) is collected and a model is trained with convolutional neural networks. The trained model takes camera images as inputs and predicts steering angles. The trained model is then used on a test track to see if a car can be driven autonomously. Please see this [report](https://github.com/kharikri/SelfDrivingCar-BehavioralCloning/blob/master/writeup_report.md) for the approach taken to implement this project.

I implemented this project in Python, Keras, and OpenCV on a quad-core CPU machine. Yes you can train DNNs on a CPU! 
