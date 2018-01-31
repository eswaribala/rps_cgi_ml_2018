# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:55:54 2017

@author: BALASUBRAMANIAM
"""


print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
'''
xvalues = np.array([0, 1, 2, 3, 4]);
yvalues = np.array([0, 1, 2, 3, 4]);
xx, yy = np.meshgrid(xvalues, yvalues)

plt.plot(xx, yy, marker='.', color='k', linestyle='none')
plt.show()

'''
#number of neighbors for each sample
n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()


print(iris.data)
X = iris.data[:,2 :4]  # we only take the first two features. We could
print(X)                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['distance']:
    # we create an instance of Neighbours Classifier
    #and fit the data.
    print(weights)
    clf = neighbors.KNeighborsClassifier\
    (n_neighbors, weights=weights)
    clf.fit(X, y)
    # save the model to disk
    import pickle
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
     
   
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    from sklearn.metrics import accuracy_score
    
    #ravel flattens n dimensional array in to row major or column major
    #np.c_ is another way of doing array concatenate
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
     
    #
    #con_mat = confusion_matrix(true_values, pred_values, [0, 1])
    #print(accuracy_score(yy,Z))


    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
   
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

    plt.show()

 # some time later...
    X = iris.data[:,2 :4] 
    y = iris.target
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X, y)
    print(result)