#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This program takes an image or video and finds the lane line on the road """


import argparse
import os

import pickle
import glob

import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip

import random


###########################
### Helper Definitions  ###
###########################

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None


class Camera_Calibrator():

    def __init__(self):

        self.objpoints = []
        self.mtx = None
        self.dist = None

    def extract_calibration_points(self,  chessboard_size = [9,6], cal_img = 'camera_cal/calibration*.jpg'):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(cal_img)

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (8,6), corners, ret)
                #write_name = 'corners_found'+str(idx)+'.jpg'
                #cv2.imwrite(write_name, img)
                cv2.imshow('img', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

    def calibrate_camera(self, img):
        # Test undistortion on an image
        img_size = (img.shape[1], img.shape[0])

        # Do camera calibration given object points and image points
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    def run_camera_calibration(self,img):
        self.extract_calibration_points()
        self.calibrate_camera(self, img)

    def get_calibration_results(self):
        return self.mtx, self.dist

    def test(self):
        self.extract_calibration_points()
        img = cv2.imread('camera_cal/calibration5.jpg')
        self.calibrate_camera(self, img)

        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        cv2.imwrite('figures/distorted.jpg',img)
        cv2.imwrite('figures/undistorted.jpg',dst)


class Perspective_Transformer():

    def __init__(self):

        self.src = None
        self.dst = None

    def set_warp_param(src,dst):
        self.src = src
        self.dst = dst


    def warp(img):
        # Compute and apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

        return warped

    def unwarp(img):
        # Compute and apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(self.dst, self.src)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

        return warped

class Thresholder():

    def __init__(self):

        self.sobel_kernel = 3

        self.abs_thresh_x = (20, 100)
        self.abs_thresh_y = (0, 255)

        self.mag_thresh = (0, 255)

        self.dir_thresh = (0.7, 1.3)

        self.hls_thresh = (170, 255)


    def abs_sobel_thresh(self, img, orient='x'):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
            abs_thresh = abs_thresh_x
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
            abs_thresh = abs_thresh_y
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= abs_thresh[0]) & (scaled_sobel <= abs_thresh[1])] = 1

        # Return the result
        return binary_output

    def mag_thresh(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= self.mag_thresh[0]) & (gradmag <= self.mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dir_threshold(self, img):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= self.dir_thresh[0]) & (absgraddir <= self.dir_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def hls_select(self,img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > hls_thresh[0]) & (s_channel <= hls_thresh[1])] = 1
        return binary_output

    def apply_all(self, image):
        gradx = self.abs_sobel_thresh(image, orient='x')
        grady = self.abs_sobel_thresh(image, orient='y')
        mag_binary = self.mag_thresh(image)
        dir_binary = self.dir_threshold(image)
        hls_binary = self.hls_select(image)

        gradient_bin = np.zeros_like(mag_binary)
        gradient_bin[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        color_bin = np.zeros_like(mag_binary)

        combined_binary = np.zeros_like(mag_binary)
        combined_binary[(gradient_bin == 1) | (color_bin == 1)] = 1

        return combined_binary

    def test(self):
        img = cv2.imread('test_images/test6.jpg')
        binary = self.apply_all(self, img)
        cv2.imwrite('figures/color.jpg',img)
        cv2.imwrite('figures/binary.jpg',binary)

class Lane_Line_Detector():

    def __init__(self):

        self.left_line = Line()
        self.right_line = Line()

        self.mtx = None
        self.dist = None

    def calibrate_camera(self,img):
        calibrator = Camera_Calibrator()
        calibrator.run_camera_calibration(img)
        self.mtx, self.dist = calibrator.get_calibration_results()


    def process_image(self, img):
        """This function takes an image, finds the lane lines, and draws them.
        
        Arguments:
            img = RGB image
        Output:
            result = RGB image where lines are drawn on lanes
        """


        return result


    def find_lanes_image(self, filepath, save_result = True):
        """This method finds laneslines on a single image file and displays it on the screen.
        
        Note: Similar to 
        Arguments:
            filepath = Path to image file
            save_result = Switch to indicate whether you want to save result
        """

        # Reading in an image. Note: mpimg.imread outputs in RGB format.
        image = mpimg.imread(filepath)

        # Printing image info, and displaying it
        print('Image at {} is now: {} with dimensions: {}'.format(filepath,type(image),image.shape))
        plt.imshow(image)
        plt.show()

        result = process_image(image)
        plt.imshow(result)
        plt.show()

        # Save Result in cv2 format (BGR)
        if save_result:
            pos = filepath.rfind('/')
            savepath = filepath[:pos] + '_output' + filepath[pos:]
            result_BGR = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            print("\n Saving image at: {}".format(savepath))
            cv2.imwrite(savepath,result_BGR)


    def find_lanes_video(self, filepath, save_result = True):
        """This method finds laneslines on a video file and displays it on the screen.
        
        Arguments:
            filepath = Path to video file
            save_result = Switch to indicate whether you want to save result
        """
        clip = VideoFileClip(filepath)
        

        result_clip = clip.fl_image(sprocess_image)

        # Save Result 
        if save_result:
            pos = filepath.rfind('/')
            savepath = filepath[:pos] + '_output' + filepath[pos:]
            print("\n Saving video at: {}".format(savepath))
            result_clip.write_videofile(savepath, audio=False)





def parse_args():
    """Supplies arguments to program"""

    parser = argparse.ArgumentParser()
    # Set video Mode
    parser.add_argument('--video-mode', dest='video_mode', action='store_true',default=False)
    # Set image Mode
    parser.add_argument('--image-mode', dest='image_mode', action='store_true',default=False)
    # Set video path if video Mode
    parser.add_argument('--video-path', dest='video_path', type=str, default="test_videos/challenge.mp4")
    # Set image path if image mode
    parser.add_argument('--image-path', dest='image_path', type=str, default="test_images/whiteCarLaneSwitch.jpg")
    # Save result mode
    parser.add_argument('--save-result', dest='save_result', action='store_true',default=True)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    if args.video_mode:
        find_lanes_video(args.video_path,save_result=args.save_result)
    elif args.image_mode:
        find_lanes_image(args.image_path,save_result=args.save_result)
    else:
        print('Enter: python lanelines.py --video-mode --video-path [PATH_TO_VIDEO] , or python lanelines.py --image-mode --image-path [PATH_TO_IMAGE]')