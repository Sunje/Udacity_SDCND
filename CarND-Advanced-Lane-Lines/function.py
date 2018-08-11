import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


class Line():
    def __init__(self):
        self.value = 0





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

    combined = np.zeros_like(hls_combined)
    combined[(gray_combined == 1) | (hls_combined == 1)] = 1
    # plt.imshow(combined, cmap='gray')
    # plt.waitforbuttonpress()

    return combined


def perspective_transform(edges):
    h,w = edges.shape[0], edges.shape[1]

    return