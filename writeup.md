# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
TODO
![alt text](https://goo.gl/c9txCC)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In the preprocessing step, I normalized the image data because I don't want some noise or outliers to affect the training and testing results. Also, I turn images to grayscale to make the data simplier, and wish to have a better result.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		  | 32x32x3 RGB image   				        | 
| Convolution 5x5   | 1x1 stride, same padding, outputs 28x28x6 |
| RELU			  |										 |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x6 				 |
| Convolution 5x5	  | 1x1 stride, same padding, outputs 10x10x16|
| RELU                    |                                                                        |
| Max pooling          | 2x2 stride,  outputs 5x5x16                           |
| Flatten                   | 400                                                                 |
| Fully connected    | 200        							          |
| RELU                     |                                                                        |
| Fully connected    | 120                                                                  |
| RELU                    |                                                                         |
| Fully connected    | 84                                                                    |
| RELU                    |                                                                         |
| Fully connected    | 43                                                                    |






#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model => 
Number of epochs: 20 
Batch size : 32
Learning rate: 0.0008
Type of optimizer: adam optimizer


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy is around 0.93~0.93 
* test set accuracy is around 0.93(sometimes 0.92)

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    I chose the LeNet architecture since the task is similar
* What were some problems with the initial architecture?
    The accuracy of testing is not high enough(0.92~0.93)
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    I added one more fully-connected layer between 400 and 120. Also, I normalized the data before doing anything. Last, I tweaked the learning rate, batch size and number of epochs.
* Which parameters were tuned? How were they adjusted and why?
    Learning rate, batch size and epochs. Just tweaked them and see the result. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    I tried to use the dropout but it did not go well, the accuracy even dropped down by a couple percents. CNN works well since we are dealing with pictures.
* How would I choose the optimizer? 
    The adam optimizer stores the exponentially decaying average of past squared gradients which make sense since it should slow down as it reaches close to the target value. It also keeps an exponentially decaying average of past gradients which similars to momentum. 
* Overfitting and Underfitting:
    I think my model is a little bit overfitting since during training, the accuracy can be as high as 0.95, but it only shows at most 0.93 on testing data. Initially, I think dropout can help me with this problem, but it ends up doing even worse. 
* How would I decide the number and type of layers?
    I just try to add layers on the LeNet model and see the result, and the point is not to make it overfitting.
* How would I tune the hyperparameter? How many values should I test and how to decide the values?
    The way is tune the model is either using for loop to track how those parameters affect the result or just simply tweaking one by one and see its result. I shoud test the values metioned above and make a graph to check the highest accuracy.
* How would I preprocess my data? Why do I need to apply a certain technique?
    I normalize and turn it into grayscale, and I mentioned the reason above. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
![alt text](https://goo.gl/iLUJE2)

I think the brightness and the angle of the traffic signs are the most crucial factor when it comes to choosing images. Both of them will deeply affect the result. Besides, background images should be taken into consideration especially the background contains other traffic signs, which will be totally confusing, even to the humans. 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The result says the accuracy is 100%, which means my model can classify:
    Children crossing
    Roundabout mandatory
    Speed limit (20km/h)
    Turn left ahead
    Yield

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 37th cell of the Ipython notebook.





