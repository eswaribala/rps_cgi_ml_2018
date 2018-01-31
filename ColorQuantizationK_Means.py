# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:15:15 2017

@author: BALASUBRAMANIAM
"""

import numpy as np
import cv2

img = cv2.imread('highcourt.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4#vary k value 

'''
temp, classified_points, means = cv2.kmeans(data=np.asarray(samples),
 K=2, bestLabels=None,
criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),
 attempts=1, 
flags=cv2.KMEANS_RANDOM_CENTERS)   #Let OpenCV choose random centers for the clusters
'''
'''
Output parameters
compactness : It is the sum of squared distance from each point to their corresponding centers.
labels : This is the label array (same as ‘code’ in previous article) where each element marked ‘0’, ‘1’.....
centers : This is array of centers of clusters.
'''
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()