import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from os.path import join, basename
from moviepy.editor import VideoFileClip
from function import find_edges, perspective_transform


def calibration():
    # Chessboard size
    nx = 9
    ny = 6

    # Read in and make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Arrays to stroe object points and image points from all the images
    objpoints = []
    imgpoints = []

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((ny*nx,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x,y coordinates
    
    for frame in images:
        # Read in each image
        img = cv2.imread(frame)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)

        # If corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            # plt.imshow(img)
            # plt.waitforbuttonpress()

    # Use cv2.calibrateCamera()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # # Check that the calibration is working well
    # img = cv2.imread('./camera_cal/calibration1.jpg')
    # undist = cv2.undistort(img,mtx,dist,None,mtx)
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # f.tight_layout()
    # ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=50)
    # ax2.imshow(undist)
    # ax2.set_title('Undistorted Image', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.waitforbuttonpress()

    return ret, mtx, dist, rvecs, tvecs


def pipeline(frame, verbose=True):

    if verbose == True:
        img = frame
    else:
        img = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)

    # Use cv2.undistort()
    undist = cv2.undistort(img,mtx,dist,None,mtx)
    edges = find_edges(undist)
    warp = perspective_transform(edges)
    # plt.imshow(edges, cmap='gray')
    # plt.waitforbuttonpress()

    return 


if __name__ =='__main__':

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibration()

    test_images = glob.glob('./test_images/*.jpg')
    for test_img in test_images:
        output = pipeline(test_img, verbose = False)
        outpath = os.path.join('output_images','output_'+basename(test_img))





