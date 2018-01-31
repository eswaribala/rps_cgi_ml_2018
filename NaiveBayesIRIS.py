# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 01:42:04 2017

@author: BALASUBRAMANIAM
"""

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = GaussianNB()
y_pred=classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)
y_pred = classifier.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"  % (iris.data.shape[0],(iris.target != y_pred).sum()))
