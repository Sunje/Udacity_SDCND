import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from os.path import join, basename
from moviepy.editor import VideoFileClip
from functions import find_edges, perspective_transform, search_fresh, search_around_poly, draw_onto_road, Line

left_line = Line()
right_line = Line()
case = None

def calibration(img_dir):
    # Chessboard size
    nx = 9
    ny = 6

    # Read in and make a list of calibration images
    images = glob.glob(img_dir)

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


def pipeline(frame, verbose = True):

    global left_line, right_line, case
    
    if verbose == True:
        img = frame
    else:
        img = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
        left_line.detected = False

    # Undistorting, finding edges, perspective transform
    undist = cv2.undistort(img,mtx,dist,None,mtx)
    # plt.imshow(undist)
    # plt.waitforbuttonpress()
    edges = find_edges(undist)
    # plt.imshow(edges)
    # plt.waitforbuttonpress()
    warped, Minv = perspective_transform(edges, case)
    # plt.imshow(warped)
    # plt.waitforbuttonpress()

    # Poly fitting the line, calculating curvature
    if left_line.detected == False:
        left_line, right_line, detected_img, line_img = search_fresh(warped, left_line, right_line, smooth = 3)
    else:
        left_line, right_line, detected_img, line_img = search_around_poly(warped, left_line, right_line, smooth = 3)
    # plt.imshow(detected_img)
    # plt.waitforbuttonpress()
    # plt.imshow(line_img)
    # plt.waitforbuttonpress()
    out_img = draw_onto_road(undist,edges,Minv,left_line,right_line, detected_img, line_img)
    # plt.imshow(out_img)
    # plt.waitforbuttonpress()
    
    return out_img


if __name__ =='__main__':

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibration(img_dir='./camera_cal/calibration*.jpg')

    # test_images = glob.glob('./test_images/*.jpg')
    # for test_img in test_images:
    #     output = pipeline(test_img, verbose = False, case = 'project_video.mp4')
    #     outpath = os.path.join('output_images','output_'+basename(test_img))
    #     cv2.imwrite(outpath,cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    test_videos = glob.glob('./test_videos/*.mp4')
    for test_video in test_videos:
        case = basename(test_video)
        outpath = os.path.join('output_videos','output_'+basename(test_video))
        clip1 = VideoFileClip(test_video, verbose = True)
        print (clip1)
        clip = clip1.fl_image(pipeline)
        clip.write_videofile(outpath,audio = False)





