# **Traffic Sign Recognition** 


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

[image1]: ./examples/train_dataset_sign_counts.png "Train dataset"
[image2]: ./examples/valid_dataset_sign_counts.png "Valid dataset"
[image3]: ./examples/test_dataset_sign_counts.png "Test dataset"
[image4]: ./examples/original_gray_normal.png "Original Gray Normal"
[image5]: ./examples/LeNet5.jpg "LeNet-5"
[image6]: ./examples/test_web_images.png "German traffic signs from the web"
[image7]: ./examples/prob_for_each_images.png "Probability for each images"
[image8]: ./examples/visualize_network_0.png "Original input"
[image9]: ./examples/visualize_network_1.png "Visualization of Network 1"
[image10]: ./examples/visualize_network_2.png "Visualization of Network 2"

---

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

* Files Submitted
  * [Traffic_Sign_Classifier.ipynb ]() is Ipython notebook with code
  * [Traffic_Sign_Classifier.ipynb ]() is HTML output of the code


* Dataset Exploration / Design and Test a Model Architecture / Test a Model on New Images
  * Please read the following contents.


---

## Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.

![alt text][image1]
![alt text][image2]
![alt text][image3]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I convert the images to grayscale since the athors of ["Traffic Sign Recognition with Multi-Scale Convolutional Networks"](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) stated that using grayscale images instead of color images improves the network's accuracy. I understand it as this way: In this problem, the form of the traffic sign is more important than its color. Of course, there is a loss of information, but considering the color is considered to be a factor that complicates the network and makes learning difficult.

As a last step, I normalized the image data to make zero mean and equal variance as described in the [class](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013).

![alt text][image4]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I carried out the project based on two model structures, LeNet-5 and AlexNet. Unfortunately, I failed to train the model based on the AlexNet structure. The structure of the two models is as follows:

* LeNet-5

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled and normalized image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU		|         									|
| Max pooling				| 2x2 stride, outputs 5x5x16        									|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x412      									|
| RELU		|         									|
|	Flatten					|		output 412										|
|	Fully Connected					|		 output 122										|
|	RELU					|		 									|
|	Dropout					|		 	keep prob 0.5								|
|	Fully Connected					|		 output 84										|
|	RELU					|		 									|
|	Dropout					|		 	keep prob 0.5								|
|	Fully Connected					|		 output 43										|
|	RELU					|		 									|

 
* AlexNet

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled and normalized image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU		|         									|
| Max pooling				| 2x2 stride, outputs 5x5x16        									|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x412      									|
| RELU		|         									|
|	Flatten					|		output 412										|
|	Fully Connected					|		 output 122										|
|	RELU					|		 									|
|	Dropout					|		 	keep prob 0.5								|
|	Fully Connected					|		 output 84										|
|	RELU					|		 									|
|	Dropout					|		 	keep prob 0.5								|
|	Fully Connected					|		 output 43										|
|	RELU					|		 									|
 
 
I expected a performance improvement by simply using the structure of the model, but it was not easy to change the internal structure(size of inner layers etc.) optimally.



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for the model architecture is in cells 6 through 10 of [Traffic_Sign_Classifier.ipynb ](). Since only the model based on the LeNet-5 structure has been successfully trained, I will only discuss this model from now on. 

![alt text][image5]

Above is the original LeNet-5 model. I added _dropout_ of _keep prob = 0.5_ to first and second fully connected layer of the model. _AdamOptimizer_ was used with _learning rate = 0.001_, _epochs = 20_, and _batch size = 128_. With slight modifications to the default model(LeNet-5) provided form the Udacity lesson, I was able to get at least 94% accuaracy.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* train set accuracy: 99.9%
* validation set accuracy: 95.4%
* test set accuracy: 93.8%
* accuracy of the German traffic sign test set from the web: 100%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * The model I used is a slightly modified version of the default model provided from the Udacity lesson. The model showed over 93% accuracy after a few trial steps, which met the project's requirement.
  
* What were some problems with the initial architecture?
  * At first, the default model had no _dropout_ layer and was not trained properly. 
  
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * After adding the _dropout_ layer, it worked really fine as it reduced overfitting problem.
  
* Which parameters were tuned? How were they adjusted and why?
  * There are hyperparameters such as _learning rate_, _epochs_, _batch size_, and _drop out probability_. To be honest, the first attempt showed 94% accuracy so I made fewer attempts to adjust hyperparameters to increase the accuracy. Instead, I tried a new attempt to set up a model based on the AlexNet structure to improve performance, but I could not optimize the internal structure(such as the inner layer size).
  
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * 

If a well known architecture was chosen:
* What architecture was chosen?
  * LeNet-5 was chosen.
  
* Why did you believe it would be relevant to the traffic sign application?
  * LeNet-5 takes 32x32x1 size image as input and solves classification problem, which has very similar structure to our project.
  
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  * I estimated the performance of the model for each sets. As mentioned above, each tests showed over 93% accuracy performance. Therefore, I decided that the model worked well.
  


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6]

As you can see these images are very regularized as the data that I used to train the model. Therefore, if the model is trained properly, it should have no difficulties to classify these images correctly.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the [Traffic_Sign_Classifier.ipynb ]().

This is the results of the prediction:

![alt text][image7]

Except for the label 15 traffic sign(no vehicles), the model could classify any other signs with 100% confidence.  


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

This is the visualization of network's feature maps:

![alt text][image8]
![alt text][image9]
![alt text][image10]


The first layer is seemed to extract some lines from the input images. However, in the case of the second layer, it is difficult to know what the layer is aiming for.
