import matplotlib.pyplot as plt
import cv2
import os
from os.path import join, basename
from moviepy.editor import VideoFileClip
from functions import line_detection, line_detection_video, line_detection_video_challenge

import globalvariables
import globalupdate


if __name__ == '__main__':

    # test on images
    test_images_dir = 'test_images'
    test_images = [join(test_images_dir, name) for name in os.listdir(test_images_dir)]
    print (test_images)

    if not os.path.exists('results'):
        folders = ['images','videos']
        for folder in folders:
            os.makedirs('results/%s'%(folder))
        
        
    for test_image in test_images:

        print('Processing image: {}'.format(test_image))
        out_path = os.path.join('results', 'images', basename(test_image))
        input_image = cv2.cvtColor(cv2.imread(test_image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        out_image = line_detection([input_image])
        cv2.imwrite(out_path, cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))

        
    plt.close('all')



    test_videos_dir = join('test_videos')
    test_videos = [join(test_videos_dir, name) for name in os.listdir(test_videos_dir)]
    print (test_videos)

    for test_video in test_videos:

        print('Processing video: {}'.format(test_video))
        print (test_video)
        out_path = os.path.join('results', 'videos', basename(test_video))
        # globalupdate.clear_history()

        if test_video == 'test_videos/challenge.mp4':
            clip1 = VideoFileClip(test_video)
            print (clip1)
            clip = clip1.fl_image(line_detection_video_challenge)
            clip.write_videofile(out_path,audio = False)

        else:    
            clip1 = VideoFileClip(test_video)
            print (clip1)
            clip = clip1.fl_image(line_detection_video)
            clip.write_videofile(out_path,audio = False)


