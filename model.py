import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline

# Flip the image and reverse the steering angle
def flip_image_and_reverse_steering_angle(image, steering):
    image = cv2.flip(image, 1)
    steering = -1*steering
    return (image, steering)

# Data collection using the provided Udacity data and augmenting that data further
DIR = "../data/data/"
lines = []
with open(DIR + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = [] #List of all images - inputs
measurements = [] # List of all steering angles - outputs
steering_correction = 0.25

for line in lines:
    # Center camera images
    center_camara_path = line[0] # Center camera path
    center_image = cv2.imread(DIR + center_camara_path)
    images.append(center_image)
    center_measurement = float(line[3]) # Center steering angle
    measurements.append(center_measurement)

    # Flip center camera images and steering angles to augment data
    flipped_image, flipped_steering = flip_image_and_reverse_steering_angle(center_image, center_measurement)
    images.append(flipped_image)
    measurements.append(flipped_steering)
 
    # Left camera images
    left_camara_path = line[1] # Left camera path
    left_image = cv2.imread(DIR + left_camara_path)
    images.append(left_image)
    left_measurement = float(line[3]) + steering_correction # Add correction so car veers to the center
    measurements.append(left_measurement)
    
    # Flip left camera images and steering angles to augment data
    flipped_image, flipped_steering = flip_image_and_reverse_steering_angle(left_image, left_measurement)
    images.append(flipped_image)
    measurements.append(flipped_steering)
    
    # Right camera images
    right_camara_path = line[2] # Right camera path
    right_image = cv2.imread(DIR + right_camara_path)
    images.append(right_image)
    right_measurement = float(line[3]) - steering_correction # Subtract correction so car veers to the center
    measurements.append(right_measurement)
    
    # Flip right camera images and steering angles to augment data
    flipped_image, flipped_steering = flip_image_and_reverse_steering_angle(right_image, right_measurement)
    images.append(flipped_image)
    measurements.append(flipped_steering)

# Train the neural network
import numpy as np

X_train = np.array(images)
y_train = np.array(measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) # Normalization
model.add(Cropping2D(cropping=((70,25),(1,1)))) # Cropping
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu')) # Convolution layer with relu activation
model.add(Dropout(.5))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(.5))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(.5))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100)) # Fully connected layer
model.add(Dropout(.5))
model.add(Dense(50))
model.add(Dropout(.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse') # Adam optimizer with mean square error loss
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.save('model.h5')