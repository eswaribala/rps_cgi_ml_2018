# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 00:13:03 2017

@author: BALASUBRAMANIAM
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
data = load_boston()
#print(data)
# Organize our data



label_names = data.target
print(label_names)

labels = data.target
print(data.feature_names)
feature_names=[]
feature_names.append(data['feature_names'][2])
'''
print(feature_names)

features = data['data']
print(features)

# Look at our data
#print(label_names)
#print('Class label = ', labels[0])
#print(feature_names)
#print(features[0])

# Split our data
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=10) # random state is seed

# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(train, train_labels)
'''
# Make predictions

#preds = gnb.predict(test)
#print(preds)

# Evaluate accuracy
#print(accuracy_score(test_labels, preds))
