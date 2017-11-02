#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run_good.mp4 a video of autonomous driving for appoimately 2 laps 
* model_accuracy.png that shows the training/validation set training accuracy

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 117-137) 

The model includes RELU layers to introduce nonlinearity (code line 120/122/124/126/128), and the data is normalized in the model using a Keras lambda layer (code line 118). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 131/133/135). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 177).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of data provided by the project, data gathered while driving in reverse direction and data gathered while driving swerving betwen lanes. The focus is the acquire data sets with balanced center/left/right side driving.

Image data are pre-processed and augemented before the training starts. The steps include applyinging random brightness, cropping and resizing the image sizes to 64x64, randomly fliping images as well as setting a minmum wheel angel threshold and adjust wheels to handle small turning angle issues usually encountered during sharp turns.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to train the modelto drive properly in straight/smooth lanes while being able to recognize the shar turns and adjust properly.

My first step was to use a convolution neural network model similar to the Nvidia model as described in the project intrduction.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with the ration of 90/10. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that the 'relu' activation layers are used on the convolution layers as well as dropout layers in between fullu converted layers. After that the overfitting problem is fixed, as can see from the model_accuracy.png included in the submission that both the training and va;idation sets follows the same trend across the training.

Then I preprocessed and augmented data as described above before feedint them to the model for training. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track because of the sharp turns. To improve the driving behavior in these cases, I adjusted the wheel angel adjustments so that the model will learn to turn more when it encounters such scenarios. The wheel angle adjustment took a few tries to get the good value for the driving track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 117-137) consisted of a convolution neural network with the following layers and layer sizes as shown in the visualization below:

The overall network architecture is show below:
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 30, 30, 24)    1824        lambda_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 30, 30, 24)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 13, 13, 36)    21636       activation_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 13, 13, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 5, 48)      43248       activation_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 5, 5, 48)      0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 3, 64)      27712       activation_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 3, 3, 64)      0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 1, 64)      36928       activation_4[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 1, 1, 64)      0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 64)            0           activation_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 80)            5200        flatten_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 80)            0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 40)            3240        dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 40)            0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 16)            656         dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 16)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            170         dropout_3[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]
====================================================================================================


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I relied on the data set provided by udacity project, the data I gathered by driving counter-clockwise as well as driving clockwise both normally and swerving between the lane. 

After the collection process,  I then preprocessed this data by applyinging random brightness, cropping and resizing the image sizes to 64x64, randomly fliping images as well as setting a minmum wheel angel threshold and adjust wheels to handle small turning angle issues usually encountered during sharp turns.

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25 as evidenced training accuracy picture in sumbission. By traing epoch 25 the model learned enough the adapt to the road condition and drive on the track without going off the road. I used an adam optimizer so that manually training the learning rate wasn't necessary. The video in the submission shows the car in autonomous mode driving 2 laps successfully. 
