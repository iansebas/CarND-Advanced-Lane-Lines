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

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        self.mtx = None
        self.dist = None

    def extract_calibration_points(self,  chessboard_size = (9,6), cal_img = 'camera_cal/calibration*.jpg'):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

        # Make a list of calibration images
        images = glob.glob(cal_img)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                #cv2.drawChessboardCorners(img, (8,6), corners, ret)
                #write_name = 'corners_found'+str(idx)+'.jpg'
                #cv2.imwrite(write_name, img)
                #cv2.imshow('img', img)
                #cv2.waitKey(500)

        #cv2.destroyAllWindows()

    def calibrate_camera(self, img):
        # Test undistortion on an image
        img_size = (img.shape[1], img.shape[0])

        # Do camera calibration given object points and image points
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size,None,None)

    def run_camera_calibration(self,img):
        self.extract_calibration_points()
        self.calibrate_camera(img)

    def get_calibration_results(self):
        return self.mtx, self.dist

    def test(self):
        self.extract_calibration_points()
        img = cv2.imread('camera_cal/calibration2.jpg')
        self.calibrate_camera(img)

        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        cv2.imwrite('figures/distorted.jpg',img)
        cv2.imwrite('figures/undistorted.jpg',dst)


class Perspective_Transformer():

    def __init__(self):

        self.src = None
        self.dst = None

    def set_warp_param(self,src,dst):
        self.src = src
        self.dst = dst


    def warp(self,img):
        # Compute and apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

        return warped

    def unwarp(self,img):
        # Compute and apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(self.dst, self.src)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

        return warped

class Thresholder():

    def __init__(self):

        self.sobel_kernel = 15

        self.abs_thresh_x = (2, 100)
        self.abs_thresh_y = (100, 255)

        self.mag_thresh = (50, 255)

        self.dir_thresh = (0.8, 1.2)

        self.h_thresh = (200, 255)
        self.l_thresh = (1, 255)
        self.s_thresh = (100, 255)


    def abs_sobel_threshold(self, channel, orient='x'):
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 1, 0))
            abs_thresh = self.abs_thresh_x
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 0, 1))
            abs_thresh = self.abs_thresh_y
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= abs_thresh[0]) & (scaled_sobel <= abs_thresh[1])] = 1

        # Return the result
        return binary_output

    def mag_threshold(self, channel):
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
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

    def dir_threshold(self, channel):
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= self.dir_thresh[0]) & (absgraddir <= self.dir_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def color_select(self,channel,hls_c):

        if hls_c == 'l':
            hls_thresh = self.l_thresh
        if hls_c == 's':
            hls_thresh = self.s_thresh
        else :
            hls_thresh = self.h_thresh

        binary_output = np.zeros_like(channel)
        binary_output[(channel > hls_thresh[0]) & (channel <= hls_thresh[1])] = 1
        return binary_output

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
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

    def apply_all(self, img):

        # we only consider edges in region of interest (roi)
        y_size = img.shape[0]
        x_size = img.shape[1]
        vertices = np.array([[(x_size*0.55,y_size*0.6),(x_size*0.45, y_size*0.6), (x_size*0.05, y_size), (x_size*0.95,y_size)]], dtype=np.int32)
        image = self.region_of_interest(img, vertices)

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        l_binary = self.color_select(l_channel,'l')
        s_binary = self.color_select(s_channel,'s')
        gradx = self.abs_sobel_threshold(s_binary*255, orient='x')
        grady = self.abs_sobel_threshold(s_binary*255, orient='y')
        mag_binary = self.mag_threshold(s_binary*255)
        dir_binary = self.dir_threshold(s_binary*255)

        gradient_bin = np.zeros_like(mag_binary)
        gradient_bin[( ( (gradx == 1) | (grady == 1)| (s_binary == 1) ) & ((dir_binary == 1) & (mag_binary == 1)) )] = 1

        combined_binary = np.zeros_like(mag_binary)
        combined_binary[(gradient_bin == 1) | (l_binary == 1)] = 1

        binary = combined_binary

        '''
        cv2.imwrite('figures/check_gradx.jpg',gradx*255)
        cv2.imwrite('figures/check_grady.jpg',grady*255)
        cv2.imwrite('figures/check_mag.jpg',mag_binary*255)
        cv2.imwrite('figures/check_dir.jpg',dir_binary*255)
        cv2.imwrite('figures/check_l_channel.jpg',l_binary*255)
        cv2.imwrite('figures/check_s_channel.jpg',s_binary*255)

        cv2.imwrite('figures/check_original.jpg',image)
        cv2.imwrite('figures/check_region.jpg',img)
        cv2.imwrite('figures/check_gradient.jpg',gradient_bin*255)
        cv2.imwrite('figures/check_binary.jpg',binary*255)
        '''

        return binary

    def test(self,img):
        binary = self.apply_all(img)
        cv2.imwrite('figures/color.jpg',img)
        binary = binary*255
        cv2.imwrite('figures/binary.jpg',binary)


class Sliding_Window_Search():

        def __init__(self):
            self.left_line = Line()
            self.right_line = Line()

            self.ploty = None
            self.img_shape = None

            # Choose the number of sliding windows
            self.nwindows = 9
            # Set the width of the windows +/- margin
            self.margin = 100
            # Set minimum number of pixels found to recenter window
            self.minpix = 50

            self.iteration = 0

        def find_curvature(self):
            # Define conversions in x and y from pixels space to meters
            ym_per_pix = 29.0/720 # meters per pixel in y dimension
            xm_per_pix = 3.0/700 # meters per pixel in x dimension

            # Fit new polynomials to x,y in world space
            plot_y = self.ploty*ym_per_pix
            left_x = self.left_line.bestx*xm_per_pix
            right_x = self.right_line.bestx*xm_per_pix
            left_fit_cr = np.polyfit(plot_y, left_x, 2)
            right_fit_cr = np.polyfit(plot_y, right_x, 2)
            # Calculate the new radii of curvature
            y_eval = np.max(plot_y)
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
            self.left_line.radius_of_curvature = left_curverad
            self.right_line.radius_of_curvature = right_curverad

            return (self.left_line.radius_of_curvature + self.right_line.radius_of_curvature)/2.0

        def find_vehicle_pos(self):
            ym_per_pix = 29.0/720 # meters per pixel in y dimension
            xm_per_pix = 3.0/700 # meters per pixel in x dimension
            vehicle_pos = self.left_line.line_base_pos*xm_per_pix
            return vehicle_pos



        def blind_histogram_search(self,binary_warped):
            # Assuming you have created a warped binary image called "binary_warped"
            self.img_shape = binary_warped.shape
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Set height of windows
            window_height = np.int(binary_warped.shape[0]/self.nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(self.nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - self.margin
                win_xleft_high = leftx_current + self.margin
                win_xright_low = rightx_current - self.margin
                win_xright_high = rightx_current + self.margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                (0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                (0,255,0), 2) 
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > self.minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > self.minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            self.left_line.allx = nonzerox[left_lane_inds]
            self.left_line.ally = nonzeroy[left_lane_inds] 
            self.right_line.allx = nonzerox[right_lane_inds]
            self.right_line.ally = nonzeroy[right_lane_inds] 

            # Fit a second order polynomial to each
            self.left_line.current_fit = np.polyfit(self.left_line.ally, self.left_line.allx, 2)
            self.right_line.current_fit = np.polyfit(self.right_line.ally, self.right_line.allx, 2)

            # Generate x and y values for plotting
            self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            self.left_line.recent_xfitted = self.left_line.current_fit[0]*self.ploty**2 + self.left_line.current_fit[1]*self.ploty + self.left_line.current_fit[2]
            self.right_line.recent_xfitted = self.right_line.current_fit[0]*self.ploty**2 + self.right_line.current_fit[1]*self.ploty + self.right_line.current_fit[2]

            self.left_line.bestx = self.left_line.recent_xfitted
            self.right_line.bestx = self.right_line.recent_xfitted

            # Calculate distance in meters of vehicle center from the line
            self.left_line.line_base_pos = np.absolute(np.mean(self.left_line.recent_xfitted)-midpoint)
            self.right_line.line_base_pos = np.absolute(np.mean(self.right_line.recent_xfitted)-midpoint)

            self.left_line.detected = True
            self.right_line.detected = True
            self.iteration += 1

        def check_for_outliers(self):

            if np.mean(self.left_line.diff) < 1 :
                self.left_line.bestx = self.left_line.recent_xfitted

            if np.mean(self.right_line.diff) < 1 :
                self.right_line.bestx = self.right_line.recent_xfitted



        def margin_search(self,binary_warped):
            # Assume you now have a new warped binary image 
            self.img_shape = binary_warped.shape
            # from the next frame of video (also called "binary_warped")
            # It's now much easier to find line pixels!
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            left_lane_inds = ((nonzerox > (self.left_line.current_fit[0]*(nonzeroy**2) + self.left_line.current_fit[1]*nonzeroy + 
            self.left_line.current_fit[2] - self.margin)) & (nonzerox < (self.left_line.current_fit[0]*(nonzeroy**2) + 
            self.left_line.current_fit[1]*nonzeroy + self.left_line.current_fit[2] + self.margin))) 

            right_lane_inds = ((nonzerox > (self.right_line.current_fit[0]*(nonzeroy**2) + self.right_line.current_fit[1]*nonzeroy + 
            self.right_line.current_fit[2] - self.margin)) & (nonzerox < (self.right_line.current_fit[0]*(nonzeroy**2) + 
            self.right_line.current_fit[1]*nonzeroy + self.right_line.current_fit[2] + self.margin)))  

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            self.left_line.current_fit = np.polyfit(lefty, leftx, 2)
            self.right_line.current_fit = np.polyfit(righty, rightx, 2)
            # Generate x and y values for plotting
            self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            self.left_line.recent_xfitted = self.left_line.current_fit[0]*self.ploty**2 + self.left_line.current_fit[1]*self.ploty + self.left_line.current_fit[2]
            self.right_line.recent_xfitted = self.right_line.current_fit[0]*self.ploty**2 + self.right_line.current_fit[1]*self.ploty + self.right_line.current_fit[2]

            self.left_line.diff = np.divide(np.absolute(self.left_line.self.left_line.bestx - self.left_line.recent_xfitted), np.absolute(self.left_line.self.left_line.bestx))*100
            self.right_line.diff = np.divide(np.absolute(self.right_line.bestx - self.right_line.recent_xfitted),np.absolute(self.right_line.self.left_line.bestx))*100

            self.check_for_outliers()

            # Calculate distance in meters of vehicle center from the line
            histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
            midpoint = np.int(binary_warped.shape[1]/2)
            self.left_line.line_base_pos = np.absolute(np.mean(self.left_line.bestx)-midpoint)
            self.right_line.line_base_pos = np.absolute(np.mean(self.right_line.bestx)-midpoint)

            self.left_line.detected = True
            self.right_line.detected = True
            self.iteration += 1



class Lane_Line_Detector():

    def __init__(self):

        self.mtx = None
        self.dist = None

        self.src_w = None
        self.dst_w = None

        self.frame = 0

    def calibrate_camera(self,img):
        calibrator = Camera_Calibrator()
        calibrator.run_camera_calibration(img)
        self.mtx, self.dist = calibrator.get_calibration_results()

    def calculate_src_and_dest(self,img):
        img_size = (img.shape[1], img.shape[0])
        src_w = np.float32([[img_size[0]*0.4, img_size[1]*0.625],[img_size[0] *0.1, img_size[1]], [img_size[0]*0.9725, img_size[1]],[img_size[0]*0.6, img_size[1]*0.625]])
        dst_w = np.float32([[(img_size[0] / 4), 0],[(img_size[0] / 4), img_size[1]],[(img_size[0] * 3 / 4), img_size[1]],[(img_size[0] * 3 / 4), 0]])
        return src_w, dst_w


    def process_image(self, img):

        preprocessor = Thresholder()
        transformer = Perspective_Transformer()
        line_searcher = Sliding_Window_Search()

        self.src_w, self.dst_w = self.calculate_src_and_dest(img)
        transformer.set_warp_param(self.src_w,self.dst_w)


        if (self.frame == 0 ):
            #Calibrate Camera
            self.calibrate_camera(img)
            
        #Undistort
        undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        # Get Binary Mask
        binary = preprocessor.apply_all(undistorted)
        #Perspective Transform
        warped = transformer.warp(binary)
        #Find Lines

        if (self.frame == 0 ):
            #Calibrate Camera
            line_searcher.blind_histogram_search(warped)
        else:
            line_searcher.blind_histogram_search(warped)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([line_searcher.left_line.bestx, line_searcher.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([line_searcher.right_line.bestx, line_searcher.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = transformer.unwarp(color_warp) 
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        curvature = line_searcher.find_curvature()
        v_pos = line_searcher.find_vehicle_pos()

        text = "Avg curvature = {} m".format(curvature)
        text2 = "Vehicle Pos from Left Lane = {}m ".format(v_pos)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result,text,(0,50), font, 1, (255,255,255),2)
        cv2.putText(result,text2,(0,100), font, 1, (255,244,255),2)

        self.frame += 1

        return result


    def find_lanes_image(self, filepath, save_result = True):
        """This method finds laneslines on a single image file and displays it on the screen.
        
        Note: Similar to 
        Arguments:
            filepath = Path to image file
            save_result = Switch to indicate whether you want to save result
        """

        # Reading in an image.
        image = cv2.imread(filepath)

        result = self.process_image(image)


        # Save Result in cv2 format (BGR)
        if save_result:
            pos = filepath.rfind('/')
            savepath = "output_files/"+ filepath[pos:]
            print("\n Saving image at: {}".format(savepath))
            cv2.imwrite(savepath,result)


    def find_lanes_video(self, filepath, save_result = True):
        """This method finds laneslines on a video file and displays it on the screen.
        
        Arguments:
            filepath = Path to video file
            save_result = Switch to indicate whether you want to save result
        """
        clip = VideoFileClip(filepath)
        

        result_clip = clip.fl_image(self.process_image)

        # Save Result 
        if save_result:
            pos = filepath.rfind('/')
            savepath = "output_files/"+ filepath[pos:]
            print("\n Saving video at: {}".format(savepath))
            result_clip.write_videofile(savepath, audio=False)





def subsystems_test():

    # Calibration
    calibrator = Camera_Calibrator()
    calibrator.test()
    mtx, dist = calibrator.get_calibration_results()

    #Pipeline

    ##Undistort
    img = cv2.imread('test_files/test2.jpg')
    img_size = (img.shape[1], img.shape[0])
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)


    ##Get Binary 
    preprocessor = Thresholder()
    preprocessor.test(undistorted)
    cv2.imwrite('figures/corrected.jpg',undistorted)
    binary = preprocessor.apply_all(undistorted)

    ##Transform
    line_searcher = Lane_Line_Detector()
    src_w, dst_w = line_searcher.calculate_src_and_dest(img)
    vrx = np.array((src_w[0],src_w[1], src_w[2],src_w[3]), np.int32)
    cv2.polylines(undistorted, [vrx], True, (0,255,255),3)
    cv2.imwrite('figures/source.jpg',undistorted)
    transformer = Perspective_Transformer()
    transformer.set_warp_param(src_w, dst_w)
    warped = transformer.warp(binary)
    cv2.imwrite('figures/warped.jpg',warped*255)


    ##Search and Fit
    line_searcher = Sliding_Window_Search()
    line_searcher.blind_histogram_search(warped)
    line_searcher.blind_histogram_search(warped)

    color_warp = np.dstack((warped*255, warped*255, warped*255))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([line_searcher.left_line.bestx, line_searcher.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([line_searcher.right_line.bestx, line_searcher.ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.imwrite('figures/dest.jpg',color_warp)


    ##Whole Process for images
    images = glob.glob("test_files/*.jpg")
    for idx, fname in enumerate(images):
        line_detector = Lane_Line_Detector()
        line_detector.find_lanes_image(fname)
    

    line_detector = Lane_Line_Detector()
    line_detector.find_lanes_video("test_files/project_video.mp4")
    




if __name__ == '__main__':
    subsystems_test()


