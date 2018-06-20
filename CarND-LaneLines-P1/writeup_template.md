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

---

#### 1. Color selection

First, select colors which represent lane lines. Through some trial and error, I set red, green, blue thershold. See _color_filter_ in [functions.py ]() for the thresholds.

![alt text][image1]

---

### 2. Gray scale

Next, transform the color selected frame to the gray scale frame.

![alt text][image1]

---

### 3. Gaussian Blur

Through Gaussian filter, blurred gray scale frame is obtained. This step improves the canny edge detection.

![alt text][image1]

---

### 4. Canny Edge Detection

This is an edge detection process. Using _cv2.Canny_ function, we find the pixel points where the gradient suddenly changes.

![alt text][image1]

---

### 5. Region of Interest

In the frame, lane lines are located in specific region. Consider only this region and discard the rest. 

![alt text][image1]

---

### 6. Region of Interest

_cv2.HoughLinesP_ function returns a set of arrays in which each elements is considered to represent the end point of the line. Then, through my _draw_line_ and _draw_line_video_ in [functions.py ](), I select the lines among the candidates (the result of _cv2.HoughLinesP_ function). 

![alt text][image1]

---

### 7. Final

Add the detected line image to the original frame.

![alt text][image1]

---

