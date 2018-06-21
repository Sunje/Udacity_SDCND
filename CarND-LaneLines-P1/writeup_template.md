# **Finding Lane Lines on the Road** 

## Lane Line Detection Pipeline

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection : There are 7 steps in my pipeline


#### 1. Color selection

First, select colors which represent the lane lines. Through trial and error, I set red, green, and blue thersholds. See _color_filter_ in [functions.py ]() for the threshold values.

![alt text][image1]


#### 2. Gray scale

Next, convert the color selected frame to a gray scale frame.

![alt text][image1]


#### 3. Gaussian Blur

Through the Gaussian filter, a blurred gray scale frame is obtained. This step improves the canny edge detection.

![alt text][image1]


#### 4. Canny Edge Detection

This is an edge detection process. Using _cv2.Canny_ function, we find the pixel points where the gradient suddenly changes.

![alt text][image1]


#### 5. Region of Interest

In the frame, the lane lines are located in specific region. Consider only this region and discard the rest. 

![alt text][image1]


#### 6. Region of Interest

_cv2.HoughLinesP_ function returns a set of arrays. Each array consisting of both ends of a straight line, ie, (x1 y1 x2 y2), is considered a candidate for a lane. Then, through my _draw_line_(_draw_line_video_) in [functions.py ](), I select the arrays that is considered a line from the candidates. The steps are follows:

1. Divides the candidate arrays left and right based on the center of the frame and analyze each.
2. Discard lines with line angles below 30 degrees and lines above 60 degrees. The result is a set of arrays that is considered a line of the current frame.
3. Combine the result of the previous frame with the result of the current frame and use the _numpy.polyfit_ function to obtain a linear function. This process allows you to draw lines smoothly. Because the previous information is used, line detection will not fail even if no lines are detected in the current frame. 
4. Plots the linear function in the specific region selected as a region of interest.

![alt text][image1]


#### 7. Final

Add the detected line image to the original frame.

![alt text][image1]

---

### 2. Identify potential shortcomings with your current pipeline

My pipeline utilizes the information from the previous frame and continues to update the information of the current frame so that it can be used in the next frame. It works well for the supplied test data. However, if the line detection is not performed continuously for several frames, there is a risk of returning line information that does not match the current line because the old line information that has not been updated is continuously used. In addition, since only straight roads are considered at present, there is a limit to properly detecting roads with curvature. It is also a problem that you need to reset the Region of Interest if the camera position changes slightly or the frame size changes slightly. It is very vulnerable to weather conditions because it is based on vision. For example, in a challenge video, there is a scene where the sun suddenly brightens the road and the line disappears. In this case, line detection is not working properly.

---

### 3. Suggest possible improvements to your pipeline

First, improve to detect road curvature. The biggest problem is when line detection fails. Considering the case where the brightness changes due to the change of weather and the disappearance of the line occurs, there is a method of measuring the brightness change in real time and optimizing the parameter values used in the _color selection_, _canny edge detection_, and _houghline_ accordingly.
