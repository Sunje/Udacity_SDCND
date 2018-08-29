import matplotlib.pyplot as plt
import cv2
import glob
import os
from os.path import join, basename
from moviepy.editor import VideoFileClip


if __name__ =='__main__':
    car_images = glob.glob('dataset/vehicles/vehicles/**/*.png')
    noncar_images = glob.glob('dataset/non-vehicles/non-vehicles/**/*.png')
    print (len(noncar_images))