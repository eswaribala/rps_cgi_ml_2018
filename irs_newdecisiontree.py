# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:01:51 2017

@author: BALASUBRAMANIAM
"""

# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# load the iris datasets
dataset = datasets.load_iris()
#print(dataset) # dispalys data and target
print(dataset.data)
print(dataset.target)
print(dataset.feature_names)

# fit a CART model to the data

model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
#print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
#print(metrics.confusion_matrix(expected, predicted))

