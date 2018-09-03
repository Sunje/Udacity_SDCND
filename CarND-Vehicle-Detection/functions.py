import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label


class Cars():
    def __init__(self):
        self.len_ppprev_bboxes = 0
        self.len_pprev_bboxes = 0
        self.len_prev_bboxes = 0
        self.curr_bboxes = []
        self.frame_count = 0

    def update_bboxes(self, bboxes):
        self.curr_bboxes.extend(bboxes)
        if self.frame_count == 3:
            self.curr_bboxes = self.curr_bboxes[len(self.curr_bboxes)//3:]
            self.len_ppprev_bboxes = self.len_pprev_bboxes
            self.len_pprev_bboxes = self.len_prev_bboxes
            self.len_prev_bboxes = len(bboxes)
        elif self.frame_count == 0:
            self.frame_count += 1
            self.len_prev_bboxes = len(bboxes)
        elif self.frame_count == 1:
            self.frame_count += 1
            self.len_pprev_bboxes = self.len_prev_bboxes
            self.len_prev_bboxes = len(bboxes)
        elif self.frame_count == 2:
            self.frame_count += 1
            self.len_ppprev_bboxes = self.len_pprev_bboxes
            self.len_pprev_bboxes = self.len_prev_bboxes
            self.len_prev_bboxes = len(bboxes)

# Code is copied from '16.Spatial Binning of Color' in class material
# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    spatial_features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return spatial_features

# Code is copied from '12.Histogram of Color' in class material
# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256), vis=False):

    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    if vis==True:
        return rhist, ghist, bhist, bin_centers, hist_features
    else:
        return hist_features

# Code is copied from '20.scikit-image HOG' in class material
# Define a function to return HOG features and visualization
def get_hog_features(img, orient=9 , pix_per_cell=8, cell_per_block=2, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        hog_features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return hog_features, hog_image
    # Otherwise call with one output
    else:      
        hog_features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return hog_features


# Code is copied from '34.Search and Classify' in class material and is slightly modified
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8,
                        cell_per_block=2, hog_channel=0,
                        get_spatial=False, get_hist=False, get_HOG=False):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_feature = []
        # Read in each one by one
        image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        else: feature_image = np.copy(image)

        if get_spatial==True:
            # Apply bin_spatial() to get spatial color features
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_feature.append(spatial_features)

        if get_hist==True:
            # Apply color_hist() also with a color space option now
            hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            file_feature.append(hist_features)

        if get_HOG==True:
            # Call get_hog_features() with vis=False, feature_vec=True
            # Append the new feature vector to the features list
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            elif hog_channel == 'gray':
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                hog_features = get_hog_features(gray, orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_feature.append(hog_features)
        
        if get_spatial==False and get_hist==False and get_HOG==False:
            print ('Error: At least one feature should be True')
            quit()

        features.append(np.concatenate(file_feature))

    # Return list of feature vectors
    return features

# Code is copied from '28.Color Classify' in class material
# Split the training and test sets, then normalize
def split_normalize(car_features, noncar_features, test_size=0.2):

    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, noncar_features)).astype(np.float64)
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            test_size=test_size, random_state=rand_state)
        # Fit a per-column scaler only on the training data
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X_train and X_test
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
        print('Feature vector length:', len(X_train[0]))
    else: 
        print('Your extract feature function only returned empty feature vectors...')

    return X_train, y_train, X_test, y_test, X_scaler

# Code is copied from '32.Sliding Window Implementation' in class material
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows

    return window_list

# Code is copied from '32.Sliding Window Implementation' in class material
# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Code is copied from '34.Seach and Classify' in class material and is slightly modified
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        get_spatial=True, get_hist=True, get_HOG=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if get_spatial == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if get_hist == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if get_HOG == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
        elif hog_channel == 'gray':
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            hog_features = get_hog_features(gray, orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Code is copied from '34.Seach and Classify' in class material
# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                    get_spatial=False, get_hist=False, get_HOG=False):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, spatial_size=spatial_size,
                    hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                    hog_channel=hog_channel, get_spatial=get_spatial, get_hist=get_hist, get_HOG=get_HOG)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(features.reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


# Code is copied from '35.Hog Sub-sampling Window Search' in class material and is slightly modified
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, svc, X_scaler, y_start_stop=[400, None], scale=1,  cspace='RGB', orient=9, cells_per_step=2,
              pix_per_cell=8, cell_per_block=2, spatial_size=(32,32), hist_bins=32, hist_range=(0,256), hog_channel=0,
              get_spatial=False, get_hist=False, get_HOG=False):

    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    bboxes = []
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    img_tosearch = draw_img[y_start_stop[0]:y_start_stop[1], :, :]

    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCR_CB)
    else: ctrans_tosearch = np.copy(img_tosearch) 
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    ch1 = ctrans_tosearch[:,:,0]

    # Compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
        hog1 = get_hog_features(ctrans_tosearch[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ctrans_tosearch[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ctrans_tosearch[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=False)
    elif hog_channel == 'gray':
        gray = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2GRAY)
        hog = get_hog_features(gray, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        hog = get_hog_features(ctrans_tosearch[:,:,hog_channel], orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1 # 105
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 # 20
    nfeat_per_block = orient*cell_per_block**2 # 36

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1 # 7
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1 # 50
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1 # 7
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            file_feature = []
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            if get_spatial==True:
                # Get spatial features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                file_feature.append(spatial_features)
            if get_hist==True:
                # Get hist features
                hist_features = color_hist(subimg, nbins=hist_bins, bins_range=hist_range)
                file_feature.append(hist_features)
            if get_HOG==True:
                # Extract HOG for this patch
                if hog_channel=='ALL':
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    file_feature.append(hog_features)
                else:
                    hog_features = hog[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    file_feature.append(hog_features)
            if get_spatial==False and get_hist==False and get_HOG==False:
                print ('Error: At least one feature should be Ture')
                quit()
            
            # Scale features and make a prediction
            # test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_features = X_scaler.transform(np.concatenate(file_feature).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append(((xbox_left, ytop_draw+y_start_stop[0]),(xbox_left+win_draw, ytop_draw+win_draw+y_start_stop[0])))
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+y_start_stop[0]),(xbox_left+win_draw,ytop_draw+win_draw+y_start_stop[0]),(0,0,255),6) 
    
    return draw_img, bboxes
    
# Code is copied from '37.Multiple Detections $ False Positives' in class material
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

# Code is copied from '37.Multiple Detections $ False Positives' in class material
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# Code is copied from '37.Multiple Detections $ False Positives' in class material
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
