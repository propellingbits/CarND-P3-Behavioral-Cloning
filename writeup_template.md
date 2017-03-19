#**Behavioral Cloning** 

##Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./WriteUpImages/p3-arch.png "Model Visualization"


## Rubric Points
### I have considered the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* p3s3.py containing the script to create and train the model
* drive5.py for driving the car in autonomous mode
* model-skinny.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive5.py file, the car can be driven autonomously around the track by executing 
```sh
python drive5.py model-skinny.h5
```

####3. Submission code is usable and readable

The p3s3.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

At last, simplicity wins again. After trying so many things and failing on each  one of them, I explored what rest of students have done. I came across the concept of skinny model architecture which gets the job done. My model consists of a convolution neural network with filter sizes of 3 and depths between 25 and 80(p3s3.py lines 18-24) 

##### Architecture
(Input images are one-channel and resized to (80w, 20h)
- 1D Convolution with a filter size of 3
- ReLU activation
- 1D MaxPooling (filter size of 2)
- Flatten
- Fully Connected Layer (64)
- Dropout with p=0.3
- Fully Connected Layer (1) -> **One final neuron for predicting one steering angle**

The model includes RELU layers to introduce nonlinearity (code line 210), and the data is normalized while preprocessing images (code line 96). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (p3s3.py 210 lines ). 

The model was trained and validated on Udacity's data set and lot of image processing is done to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (p3s3.py line 220).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Udacity's sample data is biased towards middle of road driving. It was causing car to drive off the road on turns and it was not able to make a right turn with this dataset. I balanced out dataset by reducing images with zero steering angle. I also randomly added left and right camera images as if they were taken from center camera. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I got recommendation for Nvidia's model from multiple sources. It went never really went off for me. The more I try to get it working, the more it got worse. 

After giving up multiple times, I decided to build and test my model like lego blocks.

Data augmentation is very important as what you see is what you get. If our neural network is going to see most images of one type then there are chances it is going to predict that image most of the times. 

First, I started with feeding default images from Udacity's dataset to network then I resorted to including left and right camera images, flipping, trimming, HSV conversion.

In continuation of best practices from earlier projects, I split my image and steering angle data into a training (80%) and validation set (20%). I kept my eye on training and validation loss, and accordingly modified the epochs.

The final step was to run the simulator to see how well the car was driving around track one. Inititally, car had lot of trouble while making the first turn then it had lot of trouble while making the first right turn. It will just go off the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

I did not record any new data. I used dataset provided by Udacity. 

I used left and right sides images of road as if they were taken from middle of the road so that the car can recover in case it steers off the road 

To augment the dataset, I also flipped images and angles to balance out the data set.

I shuffle the dataset as and when required and put 20% of the data into a validation set. 

I used 80% of data from dataset for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs are 6... after that there was not much change in loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
