import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import time
from os.path import join, basename
from moviepy.editor import VideoFileClip
from functions import *
from sklearn.svm import LinearSVC
import pickle
from scipy.ndimage.measurements import label

# Declare class Cars()
cars = Cars()

# Set all of the hyper-parameters
# The 'YCrCb' color space is not introduced in the original code from class material
# I refered the works of other students and add it 
cspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (32, 32)
hist_bins = 32
hist_range = (0, 256)
orient = 11 
pix_per_cell = 16 
cell_per_block = 2
hog_channel = 'ALL' # Can be 0,1,2,'ALL' or 'gray'
scale = 1.5
cells_per_step = 2

# Which mode do you want to try? Select the mode 'SH' or 'HOG' or 'ALL'
MODE = 'HOG' 

# Or you can just load the pre-trained model
# set 'True' to load the model vice-versa set 'False'
# but you must have the trained model
SKIP = True


def train_SH_SVC():
    # Extract spatial and hist features, then train a SVM classifier
    # Hyper-parameters
    global cspace, spatial_size, hist_bins, hist_range

    # 'SH' stands for 'S'patial and 'H'ist
    # Get features
    SH_car_features = extract_features(car_images, cspace=cspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, hist_range=hist_range, 
                        get_spatial=True, get_hist=True, get_HOG=False)
    SH_noncar_features = extract_features(noncar_images, cspace=cspace, spatial_size=spatial_size,
                            hist_bins=hist_bins, hist_range=hist_range, 
                            get_spatial=True, get_hist=True, get_HOG=False)

    # Get training and test sets
    SH_X_train, SH_y_train, SH_X_test, SH_y_test, SH_X_scaler = split_normalize(SH_car_features,SH_noncar_features)

    # Use a linear SVC
    SH_svc = LinearSVC()
    # Check the training time for the SH_SVC
    t=time.time()
    SH_svc.fit(SH_X_train, SH_y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SH_SVC...')
    # Check the score of the SH_SVC
    print('Test Accuracy of SH_SVC = ', round(SH_svc.score(SH_X_test, SH_y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SH_SVC predicts: ', SH_svc.predict(SH_X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', SH_y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SH_SVC')

    # Store the model
    objects = (SH_svc, SH_X_scaler)
    pickle.dump(objects, open('SH_svc.pkl', 'wb'))

    return SH_svc, SH_X_scaler

def train_HOG_SVC():

    # Extract HOG features, then train a SVM classifier
    # Hyper-parameters
    global cspace, orient, pix_per_cell, cell_per_block, hog_channel

    # Get features
    HOG_car_features = extract_features(car_images, cspace=cspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                            get_spatial=False, get_hist=False, get_HOG=True)
    HOG_noncar_features = extract_features(noncar_images, cspace=cspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                            get_spatial=False, get_hist=False, get_HOG=True)

    # Get training and test sets
    HOG_X_train, HOG_y_train, HOG_X_test, HOG_y_test, HOG_X_scaler = split_normalize(HOG_car_features, HOG_noncar_features)

    # Use a linear SVC
    HOG_svc = LinearSVC()
    # Check the training time for the HOG_SVC
    t=time.time()
    HOG_svc.fit(HOG_X_train, HOG_y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train HOG_SVC...')
    # Check the score of the HOG_SVC
    print('Test Accuracy of HOG_SVC = ', round(HOG_svc.score(HOG_X_test, HOG_y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My HOG_SVC predicts: ', HOG_svc.predict(HOG_X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', HOG_y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with HOG_SVC')

    # Store the model
    objects = (HOG_svc, HOG_X_scaler)
    pickle.dump(objects, open('HOG_svc.pkl', 'wb'))

    return HOG_svc, HOG_X_scaler

def train_ALL_SVC():

    # Extract all the features(spatial, hist, and HOG)
    # Hyper-parameters
    global cspace, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel

    # Get features
    ALL_car_features = extract_features(car_images, cspace=cspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block, hog_channel=hog_channel,
                        get_spatial=True, get_hist=True, get_HOG=True)
    ALL_noncar_features = extract_features(noncar_images, cspace=cspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block, hog_channel=hog_channel,
                        get_spatial=True, get_hist=True, get_HOG=True)

    # Get training and test sets
    ALL_X_train, ALL_y_train, ALL_X_test, ALL_y_test, ALL_X_scaler = split_normalize(ALL_car_features, ALL_noncar_features)

    # Use a linear SVC
    ALL_svc = LinearSVC()
    # Check the training time for the All_SVC
    t=time.time()
    ALL_svc.fit(ALL_X_train, ALL_y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train ALL_SVC...')
    # Check the score of the ALL_SVC
    print('Test Accuracy of ALL_SVC = ', round(ALL_svc.score(ALL_X_test, ALL_y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My ALL_SVC predicts: ', ALL_svc.predict(ALL_X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', ALL_y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with ALL_SVC')

    # Store the model
    objects = (ALL_svc, ALL_X_scaler)
    pickle.dump(objects, open('ALL_svc.pkl', 'wb'))

    return ALL_svc, ALL_X_scaler

def pipeline(img, verbose=True):

    global cars

    # Set all of the hyper-parameters
    global cspace, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, scale, cells_per_step

    bboxes = []
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # The below window parameters are copied from jaramy-shannon's work
    # The format of pair* = [[ystart,ystop],scale]
    pair1 = [[400,464],1]     
    pair2 = [[416,480],1]     
    pair3 = [[400,496],1.5]     
    pair4 = [[432,528],1.5]     
    pair5 = [[400,528],2]     
    pair6 = [[432,560],2]     
    pair7 = [[400,596],3.5]     
    pair8 = [[464,660],3.5]     
    pairs = [pair1,pair2,pair3,pair4,pair5,pair6,pair7,pair8]

    # Below is what I tried but it didn't work well
    # pair1 = [[400,464],1]
    # pair2 = [[408,472],1]
    # pair3 = [[416,480],1]
    # pair4 = [[416,512],1.5]
    # pair5 = [[432,528],1.5]
    # pair6 = [[448,544],1.5]
    # pair7 = [[432,560],2]
    # pair8 = [[448,576],2]
    # pair9 = [[464,592],2]
    # pair10 = [[448,608],2.5]
    # pair11 = [[464,624],2.5]
    # pair12 = [[496,656],2.5]
    # pairs = [pair1,pair2,pair3,pair4,pair5,pair6,pair7,pair8,pair9,pair10,pair11,pair12]
    
    window_list = []
    imcopy = np.copy(img)
    for i in range(len(pairs)):
        y_start_stop, scale = pairs[i][0], pairs[i][1]
        _, bbox = find_cars(img, svc, X_scaler, y_start_stop=y_start_stop, scale=scale, cspace=cspace,
                            orient=orient, cells_per_step=cells_per_step, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, hog_channel=hog_channel,
                            get_spatial=get_spatial, get_hist=get_hist, get_HOG=get_HOG)
        bboxes.extend(bbox)
        
        if verbose==False:
            window_list = slide_window(img,y_start_stop=y_start_stop)
            imcopy = draw_boxes(imcopy,window_list)

    # Check advanced sliding window
    if verbose==False:
        fig = plt.figure(figsize=(12,3))
        plt.subplot(121)
        plt.imshow(img)
        plt.title('Example Road Image')
        plt.subplot(122)
        plt.imshow(imcopy)
        plt.title('Advanced Sliding Window')
        plt.savefig('output_images/advanced_sliding_window.png',bbox_inches='tight')
        # plt.waitforbuttonpress()
        # plt.close()
             
    if len(bboxes) > 0:
        cars.update_bboxes(bboxes)

    heat = add_heat(heat,cars.curr_bboxes)
    heat = apply_threshold(heat,3)
    heatmap = np.clip(heat,0,255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(img, labels)
    print ('{:d} cars are detected'.format(labels[1]))

    # Just to store the test image
    if verbose == False:    
        fig = plt.figure(figsize=(12,3))
        plt.subplot(131)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(132)
        plt.imshow(heat)
        plt.title('Heat Map')
        plt.subplot(133)
        plt.imshow(labels[0])
        plt.title('Labels')
        plt.savefig('output_images/final.png',bbox_inches='tight')
        # plt.waitforbuttonpress()
        # plt.close()

    return draw_img



if __name__ =='__main__':

    # Load the data and data pre-processing
    car_images = glob.glob('dataset/vehicles/vehicles/**/*.png')
    noncar_images = glob.glob('dataset/non-vehicles/non-vehicles/**/*.png')
    n_car_images = len(car_images)
    n_noncar_images = len(noncar_images)
    ind = np.random.randint(0,n_car_images)
    image = cv2.cvtColor(cv2.imread(car_images[ind]),cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    # '''
    # Check spatial features
    spatial_features = bin_spatial(image)
    fig = plt.figure(figsize = (12,3))
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.plot(spatial_features)
    plt.title('Spatially Binned Features RGB')
    plt.savefig('output_images/spatial_visualization.png',bbox_inches='tight')
    # plt.waitforbuttonpress()
    # plt.close()
    # '''

    # '''
    # Check histogram features
    rhist, ghist, bhist, bin_centers, hist_features = color_hist(image, nbins=32, bins_range=(0, 256), vis=True)
    if rhist is not None:
        fig = plt.figure(figsize=(12,3))
        plt.subplot(141)
        plt.imshow(image)
        plt.title('Example Car Image')
        plt.subplot(142)
        plt.bar(bin_centers, rhist[0])
        plt.xlim(0, 256)
        plt.title('R Histogram')
        plt.subplot(143)
        plt.bar(bin_centers, ghist[0])
        plt.xlim(0, 256)
        plt.title('G Histogram')
        plt.subplot(144)
        plt.bar(bin_centers, bhist[0])
        plt.xlim(0, 256)
        plt.title('B Histogram')
        fig.tight_layout()
        plt.savefig('output_images/hist_visualization.png',bbox_inches='tight')
        # plt.waitforbuttonpress()
        # plt.close()
    else:
        print('Your function is returning None for at least one variable...')
    # '''

    # '''
    # Check HOG features
    hog_features, hog_image = get_hog_features(gray, vis=True)
    fig = plt.figure(figsize = (8,3))
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.savefig('output_images/hog_visualization.png',bbox_inches='tight')
    # plt.waitforbuttonpress()
    # plt.close()
    # '''

    if SKIP == False:
        if MODE == 'SH':
            svc, X_scaler = train_SH_SVC()
            get_spatial = True
            get_hist = True
            get_HOG = False
        if MODE == 'HOG':
            svc, X_scaler = train_HOG_SVC()
            get_spatial = False
            get_hist = False
            get_HOG = True
        if MODE == 'ALL':
            svc, X_scaler = train_ALL_SVC()
            get_spatial = True
            get_hist = True
            get_HOG = True
    elif SKIP == True:
        if MODE == 'SH':
            svc, X_scaler = pickle.load(open('SH_svc,pkl', 'rb'))
            get_spatial = True
            get_hist = True
            get_HOG = False
        if MODE == 'HOG':
            svc, X_scaler = pickle.load(open('HOG_svc.pkl', 'rb'))
            get_spatial = False
            get_hist = False
            get_HOG = True
        if MODE == 'ALL':
            svc, X_scaler = pickle.load(open('ALL_svc.pkl', 'rb'))
            get_spatial = True
            get_hist = True
            get_HOG = True
    else:
        print ('Error: Set parameter MODE and SKIP appropriately')
        quit()
    

    # '''
    # Check sliding window
    test_img = cv2.cvtColor(cv2.imread('test_images/test6.jpg'),cv2.COLOR_BGR2RGB)
    windows = slide_window(test_img, x_start_stop=[None, None], y_start_stop=[400, None], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    window_img = draw_boxes(test_img, windows)
    fig = plt.figure(figsize=(12,3))
    plt.subplot(121)
    plt.imshow(test_img)
    plt.title('Example Road Image')
    plt.subplot(122)
    plt.imshow(window_img)
    plt.title('Sliding Window')
    plt.savefig('output_images/sliding_window.png',bbox_inches='tight')
    # plt.waitforbuttonpress()
    # plt.close()
    # '''

    
    # '''
    # Check search window
    hot_windows = search_windows(test_img, windows, svc, X_scaler, color_space=cspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, hog_channel=hog_channel, 
                        get_spatial=get_spatial, get_hist=get_hist, get_HOG=get_HOG)
    hot_window_img = draw_boxes(test_img, hot_windows)
    fig = plt.figure(figsize=(12,3))
    plt.subplot(121)
    plt.imshow(test_img)
    plt.title('Example Road Image')
    plt.subplot(122)
    plt.imshow(hot_window_img)
    plt.title('Search window')
    plt.savefig('output_images/search_window.png',bbox_inches='tight')
    # plt.waitforbuttonpress()
    # plt.close()
    # '''
    
    # Check find cars
    test_img = cv2.cvtColor(cv2.imread('test_images/test6.jpg'),cv2.COLOR_BGR2RGB)
    find_car_img, bboxes = find_cars(test_img, svc, X_scaler, y_start_stop=[400,656], scale=scale, cspace=cspace,
                orient=orient, cells_per_step=cells_per_step, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, hog_channel=hog_channel,
                get_spatial=get_spatial, get_hist=get_hist, get_HOG=get_HOG)
    fig = plt.figure(figsize=(12,3))
    plt.subplot(121)
    plt.imshow(test_img)
    plt.title('Example Road Image')
    plt.subplot(122)
    plt.imshow(find_car_img)
    plt.title('Find Cars')
    plt.savefig('output_images/find_cars.png',bbox_inches='tight')
    # plt.waitforbuttonpress()
    # plt.close()

    # Check heat map
    heat = np.zeros_like(test_img[:,:,0]).astype(np.float)
    heat = add_heat(heat,bboxes)
    heat = apply_threshold(heat,1)
    heatmap = np.clip(heat,0,255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(test_img, labels)
    fig = plt.figure(figsize=(12,3))
    plt.subplot(131)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(132)
    plt.imshow(heatmap, cmap='hot') 
    plt.title('Heat Map')
    plt.subplot(133)
    plt.imshow(labels[0], cmap='gray')
    plt.title('Labels')
    plt.savefig('output_images/heat_map.png',bbox_inches='tight')
    # plt.waitforbuttonpress()
    # plt.close()

    # Check pipeline
    test_img = cv2.cvtColor(cv2.imread('test_images/test6.jpg'),cv2.COLOR_BGR2RGB)
    pipeline(test_img, verbose=False)
    quit()
    # Video encoding
    test_videos = glob.glob('./test_videos/*.mp4')
    for test_video in test_videos:
        case = basename(test_video)
        outpath = os.path.join('output_videos','output_'+basename(test_video))
        clip1 = VideoFileClip(test_video, verbose = True)
        clip = clip1.fl_image(pipeline)
        clip.write_videofile(outpath,audio = False)
        cars = Cars()

