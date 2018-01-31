# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 23:01:27 2017

@author: BALASUBRAMANIAM
"""

'''
Suppose you wish to know the price of a pizza. You might simply look at a menu.
This, however, is a machine learning , so we will use simple linear regression
instead to predict the price of a pizza based on an attribute of the pizza that we can
observe. Let's model the relationship between the size of a pizza and its price.

'''

import matplotlib.pyplot as plt

'''
Training Instance
'''

x = [[6], [8], [10], [14], [18]] #pizza diameter
print(x[0])
y = [[7], [9], [13], [17.5], [18]] #price in dollars

plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(x, y, 'm.') #color code is k or m or etc.,
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()

#Linear regression Algorithm to predict price
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x, y)
print ('A 11" pizza should cost: $%.2f' % model.predict(11)[0])


'''

#A 12" pizza should cost: $13.68


#multiple linear regression
# diameter and number of toppings -- 1 for training data
x = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
y = [[7], [9], [13], [17.5], [18]]
from numpy.linalg import inv
from numpy import dot, transpose
print (dot(inv(dot(transpose(x), x)), dot(transpose(x), y)))
#linear assumption between x and y
from sklearn.linear_model import LinearRegression
X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]
model = LinearRegression()
model.fit(X, y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]
predictions = model.predict(X_test)
for i, prediction in enumerate(predictions):
    print ('Predicted: %s, Target: %s' % (prediction, y_test[i]))
    print ('R-squared: %.2f' % model.score(X_test, y_test))

#polynomial assumption between x and y
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)
quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.
shape[0], 1))
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r',
linestyle='--')
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()
print (X_train)
print (X_train_quadratic)
print (X_test)
print (X_test_quadratic)
print ('Simple linear regression r-squared', regressor.score(X_test, y_test))
print ('Quadratic regression r-squared', regressor_quadratic.score(X_test_quadratic, y_test))
'''