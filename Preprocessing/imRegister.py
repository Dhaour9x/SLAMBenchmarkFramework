#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  10 17:06:20 2020

@author: Riadh dhaoui
"""
# USAGE
# python imRegister.py --first images/original_01.png --second images/modified_01.png

import cv2
import argparse
import numpy as np
import os

def registration_function(img_path1, img_path2):
    # construct the argument parse and parse the arguments
    an = argparse.ArgumentParser()
    #an.add_argument("-f", "--first", required=True,
                   # help="first input image")
    #an.add_argument("-s", "--second", required=True,
                    #help="second")
    #args = vars(an.parse_args())


    # load the two input images
    #img1_color = cv2.imread(args["first"])
    #img2_color = cv2.imread(args["second"])

    # Open the image files.
    img1_color = cv2.imread(img_path1)  # Image to be aligned.
    img2_color = cv2.imread(img_path2)  # Reference image.
    # img2_color = cv2.imread("map5 - targetMAP.jpg") # Reference image.

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    # (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    #matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    # homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    h, rigid_mask = cv2.estimateAffinePartial2D(p1, p2)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    # transformed_img = cv2.warpPerspective(img1_color, homography, (width, height))
    transformed_img = cv2.warpAffine(img1_color, h, (width, height))

    # Save the output.
    cv2.imwrite('registration_output.png', transformed_img)

    saved_image_path = os.path.join(os.getcwd(), 'registration_output.png')
    print('saved_image_path', saved_image_path)
    return saved_image_path


