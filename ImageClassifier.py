# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 01:54:46 2017

@author: BALASUBRAMANIAM
"""

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import matplotlib.image as mpimg
import numpy as np
imgs=np.array([[mpimg.imread('carrot.jpg'),mpimg.imread('butterfly.jpg')],[mpimg.imread('bus.jpg'),mpimg.imread('calendar.jpg')]])
targ=[1,2]

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(imgs)
data = imgs.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data, targ)

# Now predict the value of the digit on the second half:
expected = targ
predicted = classifier.predict(data)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))