import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        # self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        # self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_poly = None  
        #polynomial coefficients for the most recent fit
        # self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        # self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = [] 
        #y values for detected line pixels
        self.ally = [] 

        #n counts
        self.smoothing = 1
        #last fit
        self.last_bestx = None
        #new fit
        self.new_bestx = None
        #current
        self.currentx = []
        self.currenty = []
        #number of inds in previous 2 frames and current frame
        self.two_step_previous_inds_len = None
        self.one_step_previous_inds_len = None
        self.current_inds_len = None
        self.remove_inds_len = None



# Caculate directional gradient
def abs_sobel_thresh(gray, thresh_min=20, thresh_max=100):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image

    abs_sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1,0))
    abs_sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0,1))
    scaled_sobel_x = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
    scaled_sobel_y = np.uint8(255*abs_sobel_y/np.max(abs_sobel_y))
    binary_output_x = np.zeros_like(scaled_sobel_x)
    binary_output_y = np.zeros_like(scaled_sobel_y)
    binary_output_x[(scaled_sobel_x >= thresh_min) & (scaled_sobel_x <= thresh_max)] = 1
    binary_output_y[(scaled_sobel_y >= thresh_min) & (scaled_sobel_y <= thresh_max)] = 1
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # f.tight_layout()
    # ax1.imshow(binary_output_x, cmap='gray')
    # ax1.set_title('x gradient', fontsize=50)
    # ax2.imshow(binary_output_y, cmap='gray')
    # ax2.set_title('y gradient', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.waitforbuttonpress()

    return binary_output_x, binary_output_y


# Calculate gradient magnitude
def mag_thresh(gray, mag_thresh=(30,100), sobel_kernel=3):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize = sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize = sobel_kernel)
    gradmag = np.sqrt(sobel_x**2 + sobel_y**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # plt.imshow(binary_output, cmap='gray')
    # plt.waitforbuttonpress()

    return binary_output

# Calculate gradient direction
def dir_threshold(gray, threshold=(0.7,1.3), sobel_kernel=3):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize = sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize = sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= threshold[0]) & (absgraddir <= threshold[1])] = 1
    # plt.imshow(binary_output, cmap='gray')
    # plt.waitforbuttonpress()

    return binary_output

# Find edges in the grayscale frame
def gray_threshold_combined(gray):

    grad_x, grad_y = abs_sobel_thresh(gray, thresh_min=20, thresh_max=100)
    mag_binary = mag_thresh(gray, mag_thresh=(30,100), sobel_kernel=7)
    dir_binary = dir_threshold(gray, threshold=(0.7,1.3), sobel_kernel=7)
    gray_combined = np.zeros_like(dir_binary)
    gray_combined[((grad_x == 1) & (grad_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # plt.imshow(gray_combined, cmap='gray')
    # plt.waitforbuttonpress()

    return gray_combined

# Find edges in the hls frame
def hls_threshold_combined(hls):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result

    hls_h = hls[:,:,0]
    binary_output_h = np.zeros_like(hls_h)
    binary_output_h[(hls_h >= 10) & (hls_h <= 100)] = 1
    # plt.imshow(binary_output_h, cmap='gray')
    # plt.waitforbuttonpress()

    hls_s = hls[:,:,2]
    binary_output_s = np.zeros_like(hls_s)
    binary_output_s[(hls_s >= 100) & (hls_s <= 225)] = 1
    # plt.imshow(binary_output_s, cmap='gray')
    # plt.waitforbuttonpress()

    hls_combined = np.zeros_like(binary_output_s)
    hls_combined[(binary_output_h == 1) & (binary_output_s == 1)] = 1
    # plt.imshow(hls_combined, cmap='gray')
    # plt.waitforbuttonpress()

    return hls_combined


# Find edges in the frame combining all of things
def find_edges(undist):
    
    gray = cv2.cvtColor(undist,cv2.COLOR_RGB2GRAY)
    gray_combined = gray_threshold_combined(gray)
    hls = cv2.cvtColor(undist,cv2.COLOR_RGB2HLS)
    hls_combined = hls_threshold_combined(hls)

    edges = np.zeros_like(hls_combined)
    edges[(gray_combined == 1) | (hls_combined == 1)] = 1
    # plt.imshow(edges, cmap='gray')
    # plt.waitforbuttonpress()

    return edges


def perspective_transform(edges, case):
    
    h,w = edges.shape[0], edges.shape[1]

    if case == 'project_video.mp4':
        src = np.float32([[w,h-10],   # bottom right
                        [740, 460], # top right
                        [540, 460], # top left
                        [0,h-10]])  # bottom left
    elif case == 'challenge_video.mp4':
        src = np.float32([[1200,h-10], # bottom right
                        [875,500],    # top right
                        [525,500],    # top left
                        [200,h-10]])  # bottom left
    elif case == 'harder_challenge_video.mp4':
        src = np.float32([[1200,h-10], # bottom right
                        [875,500],    # top right
                        [525,500],    # top left
                        [200,h-10]])  # bottom left
    else:
        src = np.float32([[w,h-10],   # bottom right
                [740, 460], # top right
                [540, 460], # top left
                [0,h-10]])  # bottom left
                
    dst = np.float32([[w,h],      # bottom right
                      [w,0],      # top right
                      [0,0],      # top left
                      [0,h]])     # bottom left

    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(edges, M, (w,h), flags=cv2.INTER_LINEAR)
    # plt.imshow(warped)
    # plt.waitforbuttonpress()

    return warped, Minv


def sliding_window(warped, left_line, right_line, nwindows=9, margin=100, minpix=50):

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    # Set the width of the windows +/- margin
    # Set minimum number of pixels found to recenter window

    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped[warped.shape[0]//2:,:],axis=0)
    # Create an output image to draw on and visualize the result
    line_img = np.dstack((warped,warped,warped))*255
    detected_img = np.dstack((warped,warped,warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

     # Step through the windows one by one
    for window in range(nwindows):
         # Identify window boundaries in x and y (and right and left)
        win_y_bottom = warped.shape[0] - (window+1)*window_height
        win_y_top = warped.shape[0] - window*window_height
        win_xleft_left = leftx_current - margin
        win_xleft_right = leftx_current + margin
        win_xright_left = rightx_current - margin
        win_xright_right = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(detected_img,(win_xleft_left,win_y_bottom),
        (win_xleft_right,win_y_top),(0,255,0), 2) 
        cv2.rectangle(detected_img,(win_xright_left,win_y_bottom),
        (win_xright_right,win_y_top),(0,255,0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_bottom) & (nonzeroy < win_y_top) & 
        (nonzerox >= win_xleft_left) &  (nonzerox < win_xleft_right)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_bottom) & (nonzeroy < win_y_top) & 
        (nonzerox >= win_xright_left) &  (nonzerox < win_xright_right)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
    
        # If the inds > minpix, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    
    # Store current lane inds
    left_line.currentx = nonzerox[left_lane_inds]
    left_line.currenty = nonzeroy[left_lane_inds]
    right_line.currentx = nonzerox[right_lane_inds]
    right_line.currenty = nonzeroy[right_lane_inds]

    # Store length of inds
    left_line.remove_inds_len = left_line.two_step_previous_inds_len
    right_line.remove_inds_len = right_line.two_step_previous_inds_len
    left_line.two_step_previous_inds_len = left_line.one_step_previous_inds_len
    right_line.two_step_previous_inds_len = right_line.one_step_previous_inds_len
    left_line.one_step_previous_inds_len = left_line.current_inds_len
    right_line.one_step_previous_inds_len = right_line.current_inds_len
    left_line.current_inds_len = len(left_line.currentx)
    right_line.current_inds_len = len(right_line.currentx)

    # Extract left and right line pixel positions
    left_line.allx = np.append(left_line.allx, nonzerox[left_lane_inds])
    left_line.ally = np.append(left_line.ally, nonzeroy[left_lane_inds])
    right_line.allx = np.append(right_line.allx, nonzerox[right_lane_inds])
    right_line.ally = np.append(right_line.ally, nonzeroy[right_lane_inds])

    return detected_img, line_img


def calculate_curvature_and_offset(left_line,right_line, ploty, wide):

    ym_per_pix = 30/720
    xm_per_pix = 3.7/(right_line.new_bestx[-1] - left_line.new_bestx[-1])
    y_eval = 720

    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_line.new_bestx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_line.new_bestx*xm_per_pix, 2)
    
    left_line.radius_of_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5)\
                                    / np.absolute(2*left_fit_cr[0])
    right_line.radius_of_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5)\
                                     / np.absolute(2*right_fit_cr[0])
    
    left_line.line_base_pos = abs((left_line.new_bestx[-1] + right_line.new_bestx[-1])/2 - wide/2)*xm_per_pix


def search_fresh(warped, left_line, right_line, smooth = 3):

    print ("search_fresh")
    thickness = 5
    # Find our lane pixels first
    detected_img, line_img = sliding_window(warped, left_line, right_line, nwindows=9, margin=100, minpix=50)

    
    if left_line.smoothing > smooth:
        i = list(range(0,left_line.remove_inds_len))
        j = list(range(0,right_line.remove_inds_len))
        left_line.allx = np.delete(left_line.allx,i,None)
        left_line.ally = np.delete(left_line.ally,i,None)
        right_line.allx = np.delete(right_line.allx,j,None)
        right_line.ally = np.delete(right_line.ally,j,None)
        left_line.smoothing = smooth

    # Update the best fit
    left_line.best_fit_poly = np.polyfit(left_line.ally, left_line.allx, 2)
    right_line.best_fit_poly = np.polyfit(right_line.ally, right_line.allx, 2)
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_line.new_bestx = left_line.best_fit_poly[0]*ploty**2 + left_line.best_fit_poly[1]*ploty + left_line.best_fit_poly[2]
    right_line.new_bestx = right_line.best_fit_poly[0]*ploty**2 + right_line.best_fit_poly[1]*ploty + right_line.best_fit_poly[2]

    # Calculate the curvature
    calculate_curvature_and_offset(left_line, right_line, ploty, warped.shape[1])
    
    ## Visualization ##
    # Colors in the left and right lane regions
    detected_img[left_line.currenty, left_line.currentx] = [255, 0, 0]
    detected_img[right_line.currenty, right_line.currentx] = [0, 0, 255]
    line_img[left_line.currenty, left_line.currentx] = [255, 0, 0]
    line_img[right_line.currenty, right_line.currentx] = [0, 0, 255]
    # plt.imshow(detected_img)
    # plt.waitforbuttonpress()
    # plt.imshow(line_img)
    # plt.waitforbuttonpress()

    # Generate a polygon to illustrate the line and then draw the line on the image
    window_img = np.zeros_like(line_img)
    left_line_window3 = np.array([np.transpose(np.vstack([left_line.new_bestx-thickness, ploty]))])
    left_line_window4 = np.array([np.flipud(np.transpose(np.vstack([left_line.new_bestx+thickness, ploty])))])
    left_line_pts = np.hstack((left_line_window3, left_line_window4))
    right_line_window3 = np.array([np.transpose(np.vstack([right_line.new_bestx-thickness, ploty]))])
    right_line_window4 = np.array([np.flipud(np.transpose(np.vstack([right_line.new_bestx+thickness, ploty])))])
    right_line_pts = np.hstack((right_line_window3, right_line_window4))
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0,255))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,0,255))
    detected_img = cv2.addWeighted(detected_img, 1, window_img, 1, 0)
    # plt.imshow(detected_img)
    # plt.waitforbuttonpress()

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_line.new_bestx, ploty, color='yellow')
    # plt.plot(right_line.new_bestx, ploty, color='yellow')
    # plt.waitforbuttonpress()
    # plt.close()

    # Increase counts
    left_line.smoothing += 1

    # The line of the current frame is searched with 'search_fresh', 
    left_line.detected = True
    right_line.detected = True

    return left_line, right_line, detected_img, line_img

def change_integral(last,new):

    result = abs(last-new)/abs(new)

    return result

def search_around_poly(warped, left_line, right_line, smooth = 3):

    print ("search_around_poly")
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100
    thickness = 5

    # Grab activated pixels
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_line.best_fit_poly[0]*(nonzeroy**2) + left_line.best_fit_poly[1]*nonzeroy + 
                    left_line.best_fit_poly[2] - margin)) & (nonzerox < (left_line.best_fit_poly[0]*(nonzeroy**2) + 
                    left_line.best_fit_poly[1]*nonzeroy + left_line.best_fit_poly[2] + margin)))
    right_lane_inds = ((nonzerox > (right_line.best_fit_poly[0]*(nonzeroy**2) + right_line.best_fit_poly[1]*nonzeroy + 
                    right_line.best_fit_poly[2] - margin)) & (nonzerox < (right_line.best_fit_poly[0]*(nonzeroy**2) + 
                    right_line.best_fit_poly[1]*nonzeroy + right_line.best_fit_poly[2] + margin)))
    
    # Store current lane inds
    left_line.currentx = nonzerox[left_lane_inds]
    left_line.currenty = nonzeroy[left_lane_inds]
    right_line.currentx = nonzerox[right_lane_inds]
    right_line.currenty = nonzeroy[right_lane_inds]

    # Store length of inds
    left_line.remove_inds_len = left_line.two_step_previous_inds_len
    right_line.remove_inds_len = right_line.two_step_previous_inds_len
    left_line.two_step_previous_inds_len = left_line.one_step_previous_inds_len
    right_line.two_step_previous_inds_len = right_line.one_step_previous_inds_len
    left_line.one_step_previous_inds_len = left_line.current_inds_len
    right_line.one_step_previous_inds_len = right_line.current_inds_len
    left_line.current_inds_len = len(left_line.currentx)
    right_line.current_inds_len = len(right_line.currentx)

    # Extract left and right line pixel positions
    left_line.allx = np.append(left_line.allx, nonzerox[left_lane_inds])
    left_line.ally = np.append(left_line.ally, nonzeroy[left_lane_inds])
    right_line.allx = np.append(right_line.allx, nonzerox[right_lane_inds])
    right_line.ally = np.append(right_line.ally, nonzeroy[right_lane_inds])

    
    if left_line.smoothing > smooth:
        i = list(range(0,left_line.remove_inds_len))
        j = list(range(0,right_line.remove_inds_len))
        left_line.allx = np.delete(left_line.allx,i,None)
        left_line.ally = np.delete(left_line.ally,i,None)
        right_line.allx = np.delete(right_line.allx,j,None)
        right_line.ally = np.delete(right_line.ally,j,None)
        left_line.smoothing = smooth

    # Store last best fitx
    left_line.last_bestx = left_line.new_bestx
    right_line.last_bestx = right_line.new_bestx

    # Update the best fit poly and x
    left_line.best_fit_poly = np.polyfit(left_line.ally, left_line.allx, 2)
    right_line.best_fit_poly = np.polyfit(right_line.ally, right_line.allx, 2)
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_line.new_bestx = left_line.best_fit_poly[0]*ploty**2 + left_line.best_fit_poly[1]*ploty + left_line.best_fit_poly[2]
    right_line.new_bestx = right_line.best_fit_poly[0]*ploty**2 + right_line.best_fit_poly[1]*ploty + right_line.best_fit_poly[2]

    # Check the new fitted poly is reasonable when compared with the last one
    # If it is not reasonable (if the error exceed 10%), then the next frame will start the 'search_fresh'
    # The size of the integral area was compared 
    # '921600 (=1280*720)' is the entire size of the image
    # with this, the reference axis of the right line is changed to right side of the image
    if change_integral(np.trapz(left_line.last_bestx,ploty),np.trapz(left_line.new_bestx,ploty)) > 0.1 or \
       change_integral(921600-np.trapz(right_line.last_bestx,ploty),921600-np.trapz(right_line.new_bestx,ploty)) > 0.1:

        left_line.detected = False

    else:

        left_line.detected = True

    # Calculate the curvature
    calculate_curvature_and_offset(left_line, right_line, ploty, warped.shape[1])

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    detected_img = np.dstack((warped, warped, warped))*255
    line_img = np.dstack((warped, warped, warped))*255
    window_img = np.zeros_like(line_img)

    # Color in left and right line pixels
    detected_img[left_line.currenty, left_line.currentx] = [255, 0, 0]
    detected_img[right_line.currenty, right_line.currentx] = [0, 0, 255]
    line_img[left_line.currenty, left_line.currentx] = [255, 0, 0]
    line_img[right_line.currenty, right_line.currentx] = [0, 0, 255]
    

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_line.last_bestx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_line.last_bestx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_line.last_bestx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_line.last_bestx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # Draw the polygon onto the blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    detected_img = cv2.addWeighted(detected_img, 1, window_img, 0.3, 0)
    

    # Generate a polygon to illustrate the line and then draw the line on the image
    window_img = np.zeros_like(line_img)
    left_line_window3 = np.array([np.transpose(np.vstack([left_line.new_bestx-thickness, ploty]))])
    left_line_window4 = np.array([np.flipud(np.transpose(np.vstack([left_line.new_bestx+thickness, ploty])))])
    left_line_pts = np.hstack((left_line_window3, left_line_window4))
    right_line_window3 = np.array([np.transpose(np.vstack([right_line.new_bestx-thickness, ploty]))])
    right_line_window4 = np.array([np.flipud(np.transpose(np.vstack([right_line.new_bestx+thickness, ploty])))])
    right_line_pts = np.hstack((right_line_window3, right_line_window4))
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0,255))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,0,255))
    detected_img = cv2.addWeighted(detected_img, 1, window_img, 1, 0)


    # plt.imshow(detected_img)
    # plt.waitforbuttonpress()
    # plt.imshow(line_img)
    # plt.waitforbuttonpress()
    # Plot the polynomial lines onto the image
    # plt.plot(left_line.new_bestx, ploty, color='yellow')
    # plt.plot(right_line.new_bestx, ploty, color='yellow')
    

    # Increase counts
    left_line.smoothing += 1


    return left_line, right_line, detected_img, line_img


def draw_onto_road(undist,edges,Minv,left_line,right_line,detected_img,line_img):

    h,w,_ = undist.shape
    minimizing_ratio = 0.2
    m_h, m_w = int(h*minimizing_ratio), int(w*minimizing_ratio)
    x_offset, y_offset = 20, 15
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    # plt.imshow(detected_img)
    # plt.waitforbuttonpress()
    # plt.imshow(line_img)
    # plt.waitforbuttonpress()

    # Draw lines
    line_dewarped = cv2.warpPerspective(line_img, Minv, (w,h))
    out_img = cv2.addWeighted(undist, 1., line_dewarped, 1., 0)
    
    # Draw polygon between lines
    left_line_pts = np.array([np.transpose(np.vstack([left_line.new_bestx, ploty]))])
    right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_line.new_bestx, ploty])))])
    line_pts = np.hstack((left_line_pts, right_line_pts))
    road_warped = np.zeros_like(undist, dtype=np.uint8)
    cv2.fillPoly(road_warped, np.int_([line_pts]), (110,200,225))
    road_dewarped = cv2.warpPerspective(road_warped, Minv, (w,h))
    out_img = cv2.addWeighted(out_img, 1., road_dewarped, 0.5, 0)
    # plt.imshow(out_img)
    # plt.waitforbuttonpress()

    
    mask = out_img.copy()
    # Add the highlighted upper area
    highlight = cv2.rectangle(mask, pt1=(0,0), pt2=(w, m_h + y_offset*2), color = (0,102,204), thickness = cv2.FILLED)
    out_img = cv2.addWeighted(out_img, 0.8, highlight, 0.2, 0)
    # plt.imshow(out_img)
    # plt.waitforbuttonpress()

    # Add small size of edge image
    edges_mini = cv2.resize(edges, dsize=(m_w, m_h))
    edges_mini = np.dstack([edges_mini, edges_mini, edges_mini]) * 255
    out_img[y_offset:m_h + y_offset, x_offset:m_w + x_offset, :] = edges_mini

    # Add small size of detected line image in birdeye view
    detected_mini = cv2.resize(detected_img, dsize=(m_w, m_h))
    out_img[y_offset:m_h + y_offset, 2*x_offset + m_w:2*(x_offset + m_w), :] = detected_mini
    # plt.imshow(out_img)
    # plt.waitforbuttonpress()

    # Add text 'Curvature'
    mean_curvature = np.mean([left_line.radius_of_curvature, right_line.radius_of_curvature])
    cv2.putText(out_img, 'Mean curvature radius: {:02f}m'.format(mean_curvature), \
                (2*(x_offset + m_w)+x_offset,m_h//2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(out_img, 'Offset from center: {:02f}m'.format(left_line.line_base_pos), \
                (2*(x_offset + m_w)+x_offset,m_h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    # plt.imshow(out_img)
    # plt.waitforbuttonpress()

    return out_img
