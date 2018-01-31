# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:13:12 2017

@author: BALASUBRAMANIAM
"""

'''

The Iris dataset
The following four attributes of each plant were measured:
• Sepal length
• Sepal width
• Petal length
• Petal width
This is the supervised learning or classification problem
'''

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
# We load the data with load_iris from sklearn
data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']
