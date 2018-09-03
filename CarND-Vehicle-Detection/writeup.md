## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

[P5.py]: ./P5.py
[functions.py]: ./functions.py
[writeup.md]: ./writeup.md



**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/spatial_visualization.png
[image3]: ./examples/hist_visualization.png
[image4]: ./examples/hog_visualization.png
[image5]: ./examples/sliding_window.png
[image6]: ./examples/advanced_sliding_window.png
[image7]: ./examples/search_window.png
[image8]: ./examples/final.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Preview on data sets
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

### Various features 

#### 1.1 Sptial Binning of Color
The code for this step is contained in the [fucntions.py][fucntions.py] (in lines 38-44). The below image shows the extracted spatial feature from the example original image.

![alt text][image2]

By regulating the hyper-parameter(`size`), we can make a good feature extraction. This is an important factor in optimizing the model. You can regulate this parameter in the [P5.py][P5.py] (in line 21, which is named `sptial_size`).

#### 1.2 Histogram of Color
The code for this step is contained in the [fucntions.py][fucntions.py] (in lines 46-63). The below image shows the extracted hist feature from the example original image. 

![alt text][image3]

By regulating the hyper-parameters(`nbins` and `bins_range`), we can make a good feature extraction. This is an important factor in optimizing the model. You can regulate these parameters in the [P5.py][P5.py] (in lines 22-23, which are named `hist_bins` and `hist_range`, respectively).

#### 1.3 Histogram of Oriented Gradients (HOG)

The code for this step is contained in the [fucntions.py][fucntions.py] (in lines 65-82). The below image shows the extracted hog feature from the example original image.

![alt text][image4]

By regulating the hyper-parameters(`orient`, `pix_per_cell`, and `cell_per_block`), we can make a good feature extraction. This is an important factor in optimizing the model. You can regulate these parameters in the [P5.py][P5.py] (in lines 24-26).

#### 1.4 Additional parameters

In addition to the parameters listed above, there are `cspace`, `hog_channel`, and `cells_per_step`. 
* `cspace`: which determines which color space to use
* `hog_channel`: which determines what color channel to be used in hog feature extraction
* `cells_per_step`: which regulates the overlap of window in window search

You can regulate these parameters in the [P5.py][P5.py] (in lines 20 and 27-29)


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and referred the values used in other student's work. These works what I referred.
* https://github.com/jeremy-shannon/CarND-Vehicle-Detection/blob/master/vehicle_detection_project.ipynb
* https://github.com/hortovanyi/udacity-vehicle-detection-project/blob/master/Vehicle%20Detection%20Project.ipynb
* https://github.com/thomasantony/CarND-P05-Vehicle-Detection/blob/master/Project05-Training.ipynb
* https://github.com/NikolasEnt/Vehicle-Detection-and-Tracking/blob/master/VehicheDetect.ipynb

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected features

Personally, I spend a lot of time in training the model. Figuring out the best combination of features and setting the hyper-parameters were the hardest things in the project. The combinations are listed as follows:
* Train the model using the spatial and the hist features
* Train the model using the hog feature only
* Train the model using all of the features

You can select the combination in the [P5.py][P5.py] (in line 32). I used is the hog feature only combination, which achieved 98.56% accuracy in a test.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in [fucntions.py][fucntions.py] (in lines 175-219 for checking purpose, in lines 315-417 for implementation) and [P5.py][P5.py] (in lines 172-197, the scales can be found in here). I did not change the `cells_per_step` parameters.

Checking purpose
![alt text][image5]

Implementation
![alt text][image6]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The code for this step is contained in [fucntions.py][fucntions.py] (in lines 233-312) and in [P5.py][P5.py] (in lines 382-397). Ultimately I searched on two scales using YCrCb color space, all channels of HOG features, which provided a nice result.  Here is an example image:

![alt text][image7]

As mentioned above, the optimization process has been done with the determination of parameter values and what features to use.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=SYY3klg5AXY)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. The code is contained in [functions.py][functions.py] (in lines 419-451).
In addition, I declared `Cars()` class parameter `cars` in [P5.py][P5.py] (in line 15). This parameter stores the bounding boxes values of the previous 2 frames. As the position of the cars will not change drastically from frame to frame, the bounding boxes of the previous frames help to locate the car in the current frame and soften the bounding box of the car to be drawn in the image.

Here's an example result showing the heatmap, the result of `scipy.ndimage.measurements.label()`, and the resulting bounding boxes are drawn onto the last frame by thresholding the map.

![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem that I faced with this project is how to set the hyper-parameters. I spend a lot of time in regulating those parameters to find the optimized model which can find cars in the video robustly. Of course, it was not easy and I struggled a lot. To be honest, as you can see in the [result video](https://www.youtube.com/watch?v=SYY3klg5AXY), it is not that perfect. In some occasions, it fails to detect the cars. It could fail where the extracted features from the frame don't resemble those in the training dataset. `scipy.ndimage.measurements.label()` is also one of the problems that it cannot distinguish the cars when they are close to each other in the frame. The threshold value in the `apply_threshold` is also a vulnerable point that can wipe out the true positive values (the model detected cars as cars) depending on how you set this value!
As an alternative, we can use the deep learning architecture instead of machine learning. The famous [YOLO](https://pjreddie.com/darknet/yolo/) is really robust in detecting the objects and furthermore, ensures real-time, one of the important requirements of autonomous vehicle.

