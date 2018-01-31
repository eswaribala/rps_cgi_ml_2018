# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 22:35:11 2017

@author: BALASUBRAMANIAM
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
 
documents = ["This little kitty came to play when I was eating at a restaurant.",
             "Merley has the best squooshy kitten belly.",
             "saddle Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.",
             "Best cat photo I've ever taken.",
             "Climbing ninja mat.",
             "Impressed with google map feedback.",
             "Key promoter extension for toggle Chrome."]
 
'''
 tf–idf, short for term frequency–inverse document frequency,
'''
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
 
true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
 
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print
 
 
print("\n")
print("Prediction")
 
Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)
 
Y = vectorizer.transform([""])
prediction = model.predict(Y)
print(prediction)
 