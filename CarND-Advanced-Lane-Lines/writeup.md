## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---



**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[P4.py]: ./P4.py
[functions.py]: ./functions.py


[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undist.jpg "Undist Example"
[image3]: ./examples/edges.jpg "Binary Example"
[image4]: ./examples/warped.jpg "Warp Example"
[image5]: ./examples/line_img.jpg "Line Example"
[image6]: ./examples/detected_img_sliding.jpg "Sliding Window Example"
[image7]: ./examples/detected_img_poly.jpg "Around Poly Example"
[image8]: ./examples/out_img.jpg "Output Example"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the the python3 script [P4.py][P4.py] (in lines 14 through 65).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images and videos)

The code for the whole process pipeline is in the [P4.py][P4.py] (in lines 68 through 103).

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of HLS color and gradient thresholds to generate a binary image (thresholding steps at lines 48 through 132 in [functions.py][functions.py]).  Here's an example of my output for this step.
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform(edges, case)`, which appears in lines 176 through 212 in the file [functions.py][functions.py].  The `perspective_transform(edges, case)` function takes as inputs an image (`edges`), as well as argument (`case`).  I chose the hardcode the source and destination points in the following manner:

```python
h,w = edges.shape[0], edges.shape[1]

if case == 'project_video.mp4':
    src = np.float32([[w,h-10],   # bottom right
                    [740, 460],   # top right
                    [540, 460],   # top left
                    [0,h-10]])    # bottom left
elif case == 'challenge_video.mp4':
    src = np.float32([[1200,h-10], # bottom right
                    [875,500],     # top right
                    [525,500],     # top left
                    [200,h-10]])   # bottom left
elif case == 'harder_challenge_video.mp4':
    src = np.float32([[1200,h-10], # bottom right
                    [875,500],     # top right
                    [525,500],     # top left
                    [200,h-10]])   # bottom left
else:
    src = np.float32([[w,h-10],   # bottom right
            [740, 460],           # top right
            [540, 460],           # top left
            [0,h-10]])            # bottom left

dst = np.float32([[w,h],      # bottom right
                  [w,0],      # top right
                  [0,0],      # top left
                  [0,h]])     # bottom left
```

This resulted in the following source and destination points:

* for the `project_video.mp4` case

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 1280, 710     | 1280, 720     | 
| 740, 460      | 1280, 0       |
| 540, 460      | 0, 0          |
| 0, 710        | 0, 720        |

* for the `challenge_video.mp4` and `harder_challenge_video.mp4` case

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Here, various steps are applied to fit the lane line with a 2nd order polynomial. The steps are as follows:

1. For the very first time, polyfit the lane lines using the `search_fresh(warped, left_line, right_line, smooth = 3)` (at lines 330 through 395 in [functions.py][functions.py]) function for the first incoming frame. Here, the sliding window `sliding_window(warped, left_line, right_line, nwindows=9, margin=100, minpix=50)` (at lines 215 through 310 in [functions.py][functions.py]) technique is applied. As it was the very first frame to detect the lane lines, there were no previous informations regarding the lane lines. At the end, change the `left_line.detected` argument to True. This argument represent whether the lane lines have been detected. Here, I assume that the lane lines detected by using the `search_fresh` fucntion is the solid information regarding the lane lines. So the next frame will execute the `search_around_poly` function which will be discussed in the below.

2. Next, the second frame to be analyzed has the information regarding the lane lines from the previous frame. So, it doesn't need to find the lane lines from scratch. It can use the position information of the previous lane lines as a reference point to find the lane lines for the current frame. The `search_around_poly(warped, left_line, right_line, smooth = 3)` (at lines 403 through 541 in [functions.py][functions.py]) function is applied in here. An important checkpoint here is to compare the newly detected lane lines with the previous lane lines. If the difference between the two is large, it is assumed that the newly detected lane lines are not reasonable, and the `search_fresh` function is executed in the next frame by change the `left_line.detected` argument to False. The criterion for the difference between the two is 10%. That is, if the difference exceeds 10%, the above process is executed. See the lines 471 through 483 and 397 through 401 in [functions.py][function.py] for details. If there is no problem, then the next frame will also execute the `search_around_poly` function.

3. While the above processes are repeated, the arguments at lines 26 through 44 in [functions.py][functions.py] holds the acculmulated lane line informations from the previous frames. As a result, in the current frame, the lane line is detected by integrating the information of the 1 step and 2 step previous frames. This allows for smooth fitting and is based on the assumption that the lane lines up to 2 frames will not be significantly different from the lane lines in the current frame. See the lines 338 through 345 and 451 through 458 in [functions.py][functions.py] for details.

Here are some example images.

* This is the lane line image. From the warped image, the left lane line is colored red and the right lane line is colored blue.
![alt text][image5]

* This is the detected line image using `search_fresh` function.
![alt text][image6]

* This is the detected line image using `search_around_poly` function.
![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 313 through 327 in my code in [functions.py][functions.py].

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 544 through 599 in my code in [functions.py][functions.py] in the function `draw_onto_road(undist,edges,Minv,left_line,right_line,detected_img,line_img)`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/pbMKv5ClCk4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The process that I took worked really fine for the `project_video.mp4` case. But as for the `challenge_video.mp4` and espescially `harder_challenge_video.mp4` case, it did not work fine. For the `challenge_video.mp4` case, the lane lines are quite detected by simply modifying the `src` points in the `perspective_transform` function. On the otherhand, for the `harder_challenge_video.mp4` case, well... it was really hard to detect the lane lines by using the only vision information. There are so many problems such that the motorcycle intervenes in the middle blocking the lane lines; the brightness of the light changes too much, making it difficult even for a person to distinguish the lane lines; the curve is so severe that the line continuously deviates from the assumed `src` points (it could be considered to continue to deviate from the region of interest in project 1). Of course, it could improve the performance by optimizing the hyperparameters such as gradient threshold etc. However, I think that using only vision data is limited and can not provide robust results. We can use more vision data (attached at both sides underneath of the vehicle to look only at the left and the right lane line, respectively...), or we might do sensor fusion using a different sensor.
