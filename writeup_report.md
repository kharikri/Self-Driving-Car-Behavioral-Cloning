# Behavioral Cloning Project
 
## Introduction

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior or alternatively use the Udacity provided data
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

To address the above goals, I divided the project into four distinct parts which are described in detail below:
1. Data collection and augmentation
2. Data preprocessing - normalization and cropping
3. Deep neural network model architecture building and training
4. Testing on the simulator track

## Rubric Points

### Files Submitted & Code Quality 

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **writeup_report.md** summarizing the results

In addition, I included the following model file which contains a trained Nvidia model with only 600 images to get my data pipeline and model architecture correct. This took only 1 min of training on my CPU based machine for 3 epochs.
* **model_600images_1min_3e.h5**

#### 2. Submission includes functional code

Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing: 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. It also shows how data is collected and augmented before it is fed to the model. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Nvidia model architecture is used in my final submission. Later in the **Solution Design Approach** section I describe my approach in detail.

#### 2. Attempts to reduce overfitting in the model

The following is the training output using the Nvidia architecture without dropout:

```
Train on 38572 samples, validate on 9644 samples
Epoch 1/10
38572/38572 [==============================] - 1606s - loss: 0.0238 - val_loss: 0.0232
Epoch 2/10
38572/38572 [==============================] - 2017s - loss: 0.0201 - val_loss: 0.0277
Epoch 3/10
38572/38572 [==============================] - 2843s - loss: 0.0185 - val_loss: 0.0251
Epoch 4/10
38572/38572 [==============================] - 2936s - loss: 0.0172 - val_loss: 0.0242
Epoch 5/10
38572/38572 [==============================] - 3020s - loss: 0.0163 - val_loss: 0.0254
Epoch 6/10
38572/38572 [==============================] - 3379s - loss: 0.0154 - val_loss: 0.0278
Epoch 7/10
38572/38572 [==============================] - 3596s - loss: 0.0145 - val_loss: 0.0305
Epoch 8/10
38572/38572 [==============================] - 3370s - loss: 0.0139 - val_loss: 0.0265
Epoch 9/10
38572/38572 [==============================] - 3603s - loss: 0.0133 - val_loss: 0.0248
Epoch 10/10
38572/38572 [==============================] - 3522s - loss: 0.0128 - val_loss: 0.0256
```

As you notice the validation loss increases after the 4th epoch indicating overfitting. The car runs fine for most part on the test track. However, it does cross the yellow lines and runs over the red & white lines. 

To avoid overfitting, I used dropout after every convolutional layer and after some fully connected layers as shown in the following code segment:

```
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(1,1))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(.5))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(.5))**
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(.5))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(.5))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(.5))
model.add(Dense(50))
model.add(Dropout(.5))
model.add(Dense(10))
model.add(Dense(1))
```

The training output for this model is shown below:
```
Train on 38572 samples, validate on 9644 samples
Epoch 1/5
38572/38572 [==============================] - 1722s - loss: 0.0290 - val_loss: 0.0254
Epoch 2/5
38572/38572 [==============================] - 1592s - loss: 0.0250 - val_loss: 0.0252
Epoch 3/5
38572/38572 [==============================] - 1563s - loss: 0.0246 - val_loss: 0.0252
Epoch 4/5
38572/38572 [==============================] - 1559s - loss: 0.0242 - val_loss: 0.0244
Epoch 5/5
38572/38572 [==============================] - 1759s - loss: 0.0240 - val_loss: 0.0233
```

While with this model the validation loss decreased the model did not work well on the test track. The car seemed to be ramming into the walls of the bridge.  Maybe I need more epochs. 

In the following model, I increased the number of epochs to 7 and eliminated dropouts after the last two convolution layers. In the previous experiment, I had dropouts after every convolution layer. My reasoning for removing the dropouts is maybe I am getting rid of too much higher-level useful information so I wanted to preserve it.

```
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(1,1))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(.5))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(.5))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(.5))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(.5))
model.add(Dense(50))
model.add(Dropout(.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
```

My output was as follows:
```
Train on 38572 samples, validate on 9644 samples
Epoch 1/7
38572/38572 [==============================] - 1687s - loss: 0.0270 - val_loss: 0.0233
Epoch 2/7
38572/38572 [==============================] - 2165s - loss: 0.0234 - val_loss: 0.0250
Epoch 3/7
38572/38572 [==============================] - 1682s - loss: 0.0225 - val_loss: 0.0253
Epoch 4/7
38572/38572 [==============================] - 1649s - loss: 0.0220 - val_loss: 0.0238
Epoch 5/7
38572/38572 [==============================] - 1910s - loss: 0.0217 - val_loss: 0.0250
Epoch 6/7
38572/38572 [==============================] - 1572s - loss: 0.0211 - val_loss: 0.0242
Epoch 7/7
38572/38572 [==============================] - 2934s - loss: 0.0212 - val_loss: 0.0255
```

I don't know what to make out of the validation loss. It oscillates. But this model works like a charm on the test track. I held a belief that DNN is long on bag of tricks and short on theory and this proves it!

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road

For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

I followed David Silver's lecture on building and training this model. My first task was to make sure I have the entire pipeline from collecting data, training the network, generating the model and running the model on a test track was working. Once this was accomplished I experimented with various architectures. From the traffic sign classification project, I learned that convolution networks are ideal for image detection and classification. This being an image detection problem using convolution networks made sense.

I first tried with just two layers (one convolution and one fully connected layer) and then I tried the LeNet architecture of two convolution layers with subsampling and two fully connected layers. Unlike in David's videos where the car seemed to be on the track even with these simple architectures mine quickly veered away from the track. Then with the Nvidia architecture (described in the next section) my car was on the track for slightly longer duration. The car had a strong tendency to go off from the road especially at the turns. To teach the car to drive in the center of the road I used the left camera images and added a correction factor to their steering angles to induce the car to move towards the center. Similarly, I used the right camera images and subtracted a correction factor to induce the car to drive towards the center. Furthermore, I flipped all of these images left to right and the steering angles to augment the data. Track one is predominantly left turning and flipping this creates a right turning track. This data enhancement seemed to stabilize my car on the simulator.

I split my image and steering angle data into a training (80%) and validation (20%) set to see how well the model was performing. My initial models had validation errors rising after few epochs indicating overfitting. I introduced dropout after every convolution layer which reduced overfitting. While this is theoretically sound, the ultimate test is to run the model on the simulator in an autonomous mode to see if the model generalizes well. After augmenting the data and using the Nvidia architecture, I am able to go around the track-one several times.

As I have a CPU based machine to get my data collection pipeline and model architecture correct I experimented with only 600 randomly selected images out of a total of 48216 images. Using 600 images for training only took about 1 minute to produce the model for 3 epochs. I've included this model in the file called **model_600images_1min_3e.h5** file which surprisingly does quite well on the test track for more than half the lap!

#### 2. Final Model Architecture

For the final model architecture, I used the Nvidia approach. This architecture consists of 5 convolution layers followed by three fully connected layers. We flatten the network before the first fully connected layer. Relu activation is used to introduce non-linearity. For this final model the training took 4 hrs on my CPU-based laptop.

#### 3. Creation of the Training Set & Training Process

I used the Udacity supplied image data set for my training. The original data set consisted of 24108 images. After augmentation by flipping the images I doubled the dataset to 48216 images which was just about the number of images my CPU based computer (quad-core CPU with 8GB of memory) could handle without running out of memory.

I built a simple pipeline to collect my image dataset into a list as shown below:
```
for line in lines:
    #Center camera images
    center_camara_path = line[0] # Center camera path
    center_image = cv2.imread(DIR + center_camara_path)
    images.append(center_image)
    center_measurement = float(line[3]) # Center steering angle
    measurements.append(center_measurement)

    #Flip center camera images and steering angles to augment data
    flipped_image, flipped_steering = flip_image_and_reverse_steering_angle(center_image, center_measurement)
    images.append(flipped_image)
    measurements.append(flipped_steering)
 
    #Left camera images
    left_camara_path = line[1] # Left camera path
    left_image = cv2.imread(DIR + left_camara_path)
    images.append(left_image)
    left_measurement = float(line[3]) + steering_correction # Add correction so car veers to the center
    measurements.append(left_measurement)
    
    #Flip left camera images and steering angles to augment data
    flipped_image, flipped_steering = flip_image_and_reverse_steering_angle(left_image, left_measurement)
    images.append(flipped_image)
    measurements.append(flipped_steering)
    
    #Right camera images
    right_camara_path = line[2] # Right camera path
    right_image = cv2.imread(DIR + right_camara_path)
    images.append(right_image)
    right_measurement = float(line[3]) - steering_correction # Subtract correction so car veers to the center
    measurements.append(right_measurement)
    
    #Flip right camera images and steering angles to augment data
    flipped_image, flipped_steering = flip_image_and_reverse_steering_angle(right_image, right_measurement)
    images.append(flipped_image)
    measurements.append(flipped_steering)
 ```
First I collect the center camera images and then I augment these images by flipping them. I repeat this process for the left camera and the right camera images. Furthermore, I add a correction factor of 0.25 to the steering angle of the left image to simulate the effect of steering back to center from left. For the right camera image, I subtract a correction factor of 0.25 to simulate the effect of steering back to center from right. This is a very inexpensive way to train the car to drive on the track. The alternative is to collect the data on the track which is time consuming. 

The following pictures illustrate some of the data augmentation and preprocessing methods used. These two pictures show center camera image and its flipped image to augment the data. Track one is predominantly left turning. Flipping the image creates a right turning track.

![alt text](file:///C:/Users/Kris/Desktop/center_2016_12_01_13_30_48_287.jpg)
![alt text](file:///C:/Users/Kris/Desktop/flipped_center_2016_12_01_13_30_48_287.jpg)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Center Camera Image &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Flipped Center Camera Image

In the cropped picture below we crop the top 70 pixels and the bottom 25 pixels to eliminate the noise such as sky, trees, dashboard, steering wheel etc. Cropping has a nice additional property of reducing the size of the data.

![alt text](https://github.com/kharikri/SelfDrivingCar-BehavioralCloning/blob/master/Images/center_2016_12_01_13_30_48_287.jpg)
![alt text](file:///C:/Users/Kris/Desktop/cropped_center_2016_12_01_13_30_48_287.jpg)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Center Camera Image &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Cropped Center Camera Image

Besides cropping I normalized the data from 0 to 255 to -0.5 to 0.5. Neural networks work well with small numbers. Both the preprocessing steps (cropping and normalization) were done inside the model so during the simulation the same preprocessing is done in the autonomous mode.

I finally randomly shuffled the data set and put 20% of the data into a validation set and rest into the training set. 

I used this training data for training the model. The validation set in theory should be used to determine if the model is over or under fitting. As mentioned before, in my final training run the validation error oscillates but the model performs well on the test track. I used the adam optimizer so that manually training the learning rate wasn't necessary.


### Simulation

#### 1. Navigation of the car on the simulator in an autonomous mode

The car can be driven autonomously on the test track using the generated model (model.h5) with the following command:
```sh
python drive.py model.h5
```

A video of the car running autonomously around the track for little more than two laps can be seen [here](https://???). I used 800x600 resolution on the simulator for this video. 


## Summary
My overall take on the project is it is fairly straightforward to collect, augment, and preprocess data, build and train the model and test the model on the simulator but time consuming to perform various experiments to make sure the model generalizes well to the test track. These experiments included data augmentation methods, model architecture selection, architecture fine tuning (ex: relu vs elu, dropouts, etc), optimal number of epochs for training, and testing on the track. I really enjoyed this project as it gave me a good idea of the complexities and bottlenecks of deep neural networks.

I'd like to thank David Silver for his excellent lecture in explaining this project which helped me implement it.
