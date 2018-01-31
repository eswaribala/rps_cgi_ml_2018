# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 00:57:47 2017

@author: BALASUBRAMANIAM
"""

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
 
#%pylab inline
#sns.set()
 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
 
data = pd.read_csv("Wholesale customers data.csv")
data.drop(["Channel", "Region"], axis = 1, inplace = True)
 
data = data[["Grocery", "Milk"]]
data = data.as_matrix().astype("float32", copy = False)
 
stscaler = StandardScaler().fit(data)
data = stscaler.transform(data)
 
plt.scatter(data[:,0], data[:,1])
plt.xlabel("Groceries")
plt.ylabel("Milk")
plt.title("Wholesale Data - Groceries and Milk")
plt.savefig("results/wholesale.png", format = "PNG")
 
dbsc = DBSCAN(eps = .5, min_samples = 15).fit(data)
 
labels = dbsc.labels_
core_samples = np.zeros_like(labels, dtype = bool)
core_samples[dbsc.core_sample_indices_] = True
 
unique_labels = np.unique(labels)
colors = plt.cm.Spectral(np.linspace(0,1, len(unique_labels)))
 
for (label, color) in zip(unique_labels, colors):
    class_member_mask = (labels == label)
    xy = data[class_member_mask & core_samples]
    plt.plot(xy[:,0],xy[:,1], 'o', markerfacecolor = color, markersize = 10)
    
    xy2 = data[class_member_mask & ~core_samples]
    plt.plot(xy2[:,0],xy2[:,1], 'o', markerfacecolor = color, markersize = 5)
plt.title("DBSCAN on Wholsesale data")
plt.xlabel("Grocery (scaled)")
plt.ylabel("Milk (scaled)")
plt.savefig("results/dbscan_wholesale.png", format = "PNG")