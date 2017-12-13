#!/usr/bin/env python

import cv2
import numpy as np
import glob
import os

#img_scene = cv2.imread('/home/turtlebot/catkin_ws/src/project/src/detections/image0.png', 1)  # Scene Image

def identify_image(img_scene):
	files = glob.glob('/home/turtlebot/catkin_ws/src/project/src/weapons/*.png')  # Object Images
	detector = cv2.ORB(nfeatures = 2000)  #Initialise the detector
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Initialise the matcher

	object_array = []  # Empty list to store objects
    for myFile in files:
        img_object = cv2.imread(myFile, 1)  # Read each object image
        string = os.path.basename(myFile)  # Read file name of object image
        string = os.path.splitext(string)[0]  # Split off .png from file name
        object_array.append((img_object, string))  # Add object image and name to end of list

	sim_array = []  # Empty list to store data for each object image
	for (img_object, string) in object_array:
		kp_object, des_object = detector.detectAndCompute(img_object, None)  # Find all keypoints and descriptors of object image
		kp_scene, des_scene = detector.detectAndCompute(img_scene, None)  # Find all keypoints and descriptors of scene image

		matches = matcher.knnMatch(des_object, des_scene, k=2)  # Match all descriptors together

		good = []  # Empty list to store only the good matches
		for m,n in matches:  # Lowe's Ratio Test - Ratio of distance from closest neighbour to second closest
			if m.distance < 0.75*n.distance:  # 0.75 can be tuned
				good.append(m)  # Add good matches to end of list
		sim_array.append((string, good, img_object, good, kp_object, kp_scene))  # Add the name, matches, object image, object keypoints and scene keypoints to list

	sim_array = sorted(sim_array, key= lambda x: len(x[1]), reverse=True)  # Sort the list by how many elements in good matches list (largest first)
	print sim_array[0][0]  # Print name of object image with the most good matches

	src_pts = np.float32([ sim_array[0][4][m.queryIdx].pt for m in sim_array[0][1] ]).reshape(-1,1,2)  # Get the object image keypoints from the good matches
	dst_pts = np.float32([ sim_array[0][5][m.trainIdx].pt for m in sim_array[0][1] ]).reshape(-1,1,2)  # Get the scene image keypoints from the good matches

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)  # Find the homography matrix relating the keypoints from each image

	h, w = sim_array[0][2].shape[:2]  # Retrieve the dimensions of the object image

	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)  # Calculate all four corners of the object image
	dst = cv2.perspectiveTransform(pts,M)  # Apply a perspective transform to the corners of the object image corresponding to the homography matrix

	cv2.polylines(img_scene,[np.int32(dst)],True,(0,0,255),3, cv2.CV_AA)  # Draw a polygon using the newly transformed corner points onto the scene image

	#cv2.imshow('scene', img_scene)  # Show image for convenience and verification
	return img_scene
	'''
	while(True):
		if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for q to be pressed
			break

	cv2.destroyAllWindows()  # Destroy all windows before closing'''
