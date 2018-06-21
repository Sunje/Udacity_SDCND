import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

import globalvariables
import globalupdate


def color_filter(img, red_threshold = 200, green_threshold = 100, blue_threshold = 50):
    color_select = np.copy(img)
    threshold = (img[:,:,0] < red_threshold) |\
                (img[:,:,1] < green_threshold) |\
                (img[:,:,2] < blue_threshold)
    color_select[threshold] = [0,0,0]
    # plt.imshow(color_select)
    # plt.waitforbuttonpress()
    return color_select

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, vertices, color=[255, 0, 0], thickness=2):
    
    img_x = np.shape(img)[1]/2
    left = lines[lines[:,0,0] < img_x]
    right = lines[lines[:,0,0] > img_x]

    i = 0
    left_x = None
    for leftline in left:
        for x1,y1,x2,y2 in leftline:
            if (x2-x1) == 0:
                pass

            elif (y1-y2)/(x2-x1) > math.tan(30*math.pi/180) and (y1-y2)/(x2-x1) < math.tan(60*math.pi/180):

                if i ==0:
                    left_x = np.array([x1,x2])
                    left_y = np.array([y1,y2])
                    i += 1
                else:
                    left_x = np.append(left_x,[x1,x2])
                    left_y = np.append(left_y,[y1,y2])
            else:
                pass


    i = 0
    right_x = None
    for rightline in right:
        for x1,y1,x2,y2 in rightline:
            if (x2-x1) == 0:
                pass

            elif (y2-y1)/(x2-x1) > math.tan(30*math.pi/180) and (y2-y1)/(x2-x1) < math.tan(60*math.pi/180):

                if i ==0:
                    right_x = np.array([x1,x2])
                    right_y = np.array([y1,y2])
                    i += 1
                else:
                    right_x = np.append(right_x,[x1,x2])
                    right_y = np.append(right_y,[y1,y2])
            else:
                pass

    left_poly = np.polyfit(left_x,left_y,1)
    left_yl = vertices[0,0,1]
    left_yr = vertices[0,1,1] + 30
    left_xl = ((left_yl - left_poly[1])/left_poly[0]).astype(np.int32)
    left_xr = ((left_yr - left_poly[1])/left_poly[0]).astype(np.int32)

    right_poly = np.polyfit(right_x,right_y,1)
    right_yl = vertices[0,2,1] + 30
    right_yr = vertices[0,3,1]
    right_xl = ((right_yl - right_poly[1])/right_poly[0]).astype(np.int32)
    right_xr = ((right_yr - right_poly[1])/right_poly[0]).astype(np.int32)

    cv2.line(img, (left_xl,left_yl), (left_xr,left_yr),color,thickness = 10)
    cv2.line(img, (right_xl,right_yl), (right_xr,right_yr),color,thickness = 10)



def draw_lines_video(img, lines, vertices, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    img_x = np.shape(img)[1]/2
    left = lines[lines[:,0,0] < img_x]
    right = lines[lines[:,0,0] > img_x]

    i = 0
    left_x = None
    for leftline in left:
        for x1,y1,x2,y2 in leftline:
            if (x2-x1) == 0:
                pass

            elif (y1-y2)/(x2-x1) > math.tan(30*math.pi/180) and (y1-y2)/(x2-x1) < math.tan(60*math.pi/180):

                if i ==0:
                    left_x = np.array([x1,x2])
                    left_y = np.array([y1,y2])
                    i += 1
                else:
                    left_x = np.append(left_x,[x1,x2])
                    left_y = np.append(left_y,[y1,y2])
            else:
                pass


    i = 0
    right_x = None
    for rightline in right:
        for x1,y1,x2,y2 in rightline:
            if (x2-x1) == 0:
                pass

            elif (y2-y1)/(x2-x1) > math.tan(30*math.pi/180) and (y2-y1)/(x2-x1) < math.tan(60*math.pi/180):

                if i ==0:
                    right_x = np.array([x1,x2])
                    right_y = np.array([y1,y2])
                    i += 1
                else:
                    right_x = np.append(right_x,[x1,x2])
                    right_y = np.append(right_y,[y1,y2])
            else:
                pass

    if left_x is None:
        left_X = np.append([],globalvariables.previous_left_x)
        left_Y = np.append([],globalvariables.previous_left_y)
    else:
        left_X = np.append(left_x,globalvariables.previous_left_x)
        left_Y = np.append(left_y,globalvariables.previous_left_y)
        globalupdate.update_left(left_x,left_y)

    left_poly = np.polyfit(left_X,left_Y,1)
    left_yl = vertices[0,0,1]
    left_yr = vertices[0,1,1] + 30
    left_xl = ((left_yl - left_poly[1])/left_poly[0]).astype(np.int32)
    left_xr = ((left_yr - left_poly[1])/left_poly[0]).astype(np.int32)

    
    if right_x is None:
        right_X = np.append([],globalvariables.previous_right_x)
        right_Y = np.append([],globalvariables.previous_right_y)
    else:

        right_X = np.append(right_x,globalvariables.previous_right_x)
        right_Y = np.append(right_y,globalvariables.previous_right_y)
        globalupdate.update_right(right_x,right_y)

    right_poly = np.polyfit(right_X,right_Y,1)
    right_yl = vertices[0,2,1] + 30
    right_yr = vertices[0,3,1]
    right_xl = ((right_yl - right_poly[1])/right_poly[0]).astype(np.int32)
    right_xr = ((right_yr - right_poly[1])/right_poly[0]).astype(np.int32)

    cv2.line(img, (left_xl,left_yl), (left_xr,left_yr),color,thickness = 10)
    cv2.line(img, (right_xl,right_yl), (right_xr,right_yr),color,thickness = 10)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, vertices, verbose = True):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is None:
        lines = globalvariables.previous_lines
    else:
        globalupdate.update_line(lines)
 
    if verbose == False:
        draw_lines(line_img, lines, vertices)
    else:
        draw_lines_video(line_img, lines, vertices)
    return line_img



# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def line_detection(frames):
    
    img_x = frames[0].shape[1]
    img_y = frames[0].shape[0]
    vertices = np.array([[(50,img_y),(400, 300), (560, 300), (950,img_y)]])
    verbose = False
    
    for t in range(0, len(frames)):

        color_selected = color_filter(frames[t])
        # plt.imshow(color_selected)
        # plt.waitforbuttonpress()
        # plt.savefig('color_selected.jpg')
        img_gray = grayscale(color_selected)
        # plt.imshow(img_gray)
        # plt.waitforbuttonpress()
        # plt.savefig('img_gray.jpg')
        blur_gray = gaussian_blur(img_gray,5)
        # plt.imshow(blur_gray)
        # plt.waitforbuttonpress()
        # plt.savefig('blur_gray.jpg')
        edges = canny(blur_gray,50,80)
        # plt.imshow(edges)
        # plt.waitforbuttonpress()
        # plt.savefig('edges.jpg')
        masked_edges = region_of_interest(edges,vertices)
        # plt.imshow(masked_edges)
        # plt.waitforbuttonpress()
        # plt.savefig('masked_edges.jpg')
        lines = hough_lines(masked_edges,2,np.pi/180,60,120,100,vertices,verbose)
        # plt.imshow(lines)
        # plt.waitforbuttonpress()
        # plt.savefig('lines.jpg')
        lines_edges = weighted_img(lines,frames[t])
        # plt.imshow(lines_edges)
        # plt.waitforbuttonpress()
        # plt.savefig('lines_edges.jpg')
        
    return lines_edges


def line_detection_video(frames):

    img_x = frames.shape[1]
    img_y = frames.shape[0]
    vertices = np.array([[(50,img_y),(400, 300), (560, 300), (950,img_y)]])
    verbose = True

    color_selected = color_filter(frames)
    img_gray = grayscale(color_selected)
    blur_gray = gaussian_blur(img_gray,5)
    edges = canny(blur_gray,50,80)
    masked_edges = region_of_interest(edges,vertices)
    lines = hough_lines(masked_edges,2,np.pi/180,60,120,100,vertices,verbose)
    lines_edges = weighted_img(lines,frames)

    return lines_edges


def line_detection_video_challenge(frames):

    img_x = frames.shape[1]
    img_y = frames.shape[0]
    vertices = np.array([[(100,img_y),(540, 450), (740, 450), (1180,img_y)]])
    verbose = True

    color_selected = color_filter(frames)
    img_gray = grayscale(color_selected)
    blur_gray = gaussian_blur(img_gray,5)
    edges = canny(blur_gray,50,80)
    masked_edges = region_of_interest(edges,vertices)
    lines = hough_lines(masked_edges,2,np.pi/180,60,120,100,vertices,verbose)
    lines_edges = weighted_img(lines,frames)

    return lines_edges
