# Introduction
In this project, I trained and validated a Convolutional Neural Network to classify 43 traffic signs using [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I tried out the German traffic signs from the web.

# The goals / steps of this project:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


### Basic Summary of the Data Set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. 
![alt text](https://goo.gl/c9txCC)

# Design and Test a Model Architecture

### Preprocessing the Data
In the preprocessing step, I normalized the image data because I don't want some noise or outliers to affect the training and testing results. Also, I turn images to grayscale to make the data simplier, and wish to have a better result.

### Model Architecture

My final model consisted of the following layers:

| Layer                 |     Description                                | 
|:---------------------:|:---------------------------------------------:| 
| Input                   | 32x32x3 RGB image                           | 
| Convolution 5x5   | 1x1 stride, same padding, outputs 28x28x6 |
| RELU              |                                         |
| Max pooling                | 2x2 stride,  outputs 14x14x6                  |
| Convolution 5x5      | 1x1 stride, same padding, outputs 10x10x16|
| RELU                    |                                                                        |
| Max pooling          | 2x2 stride,  outputs 5x5x16                           |
| Flatten                   | 400                                                                 |
| Fully connected    | 200                                              |
| RELU                     |                                                                        |
| Fully connected    | 120                                                                  |
| RELU                    |                                                                         |
| Fully connected    | 84                                                                    |
| RELU                    |                                                                         |
| Fully connected    | 43                                                                    |

### Details of How I Train the Model
Number of epochs: _20_
Batch size : _32_
Learning rate: _0.0007_
Type of optimizer: _Adam optimizer_


# Result
My final model results were:
* validation set accuracy is around 0.93~0.93 
* test set accuracy is around 0.93(sometimes 0.92)
### Test a Model on 5 New Images
![5 New Images](https://goo.gl/iLUJE2)
The result says the accuracy is `100%`, which means my model can classify:
* Children crossing
* Roundabout mandatory
* Speed limit (20km/h)
* Turn left ahead
* Yield






