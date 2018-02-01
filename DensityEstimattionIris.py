# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 20:28:44 2017

@author: BALASUBRAMANIAM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import seaborn as sns;
#mean, cov = [0, 2], [(1, .5), (.5, 1)]
#x, y = np.random.multivariate_normal(mean, cov, size=50).T
#ax = sns.kdeplot(x)
#Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome

diabetics = pd.read_csv('diabetes.csv')
print(diabetics.keys())
#iris = sns.load_dataset("iris")
#print(iris.species)
my_data_frame = pd.DataFrame(diabetics)
#print(my_data_frame.head())

#p=plt.hist(my_data_frame.sepal_length)
#sns.set(style="ticks", color_codes=True) # change style
'''
g = sns.pairplot(iris, hue="species")

g = sns.pairplot(iris, kind="reg", hue="species")
'''
#setosa = iris.loc[iris.species == "setosa"]
#virginica = iris.loc[iris.species == "virginica"]
#versicolor=iris.loc[iris.species == "versicolor"]

positive=diabetics.loc[diabetics.Outcome==1]
negative=diabetics.loc[diabetics.Outcome==0]
ax = sns.kdeplot(positive.BMI, positive.Glucose,
                  cmap="Greens", shade=True, shade_lowest=False)
ax = sns.kdeplot(negative.Pregnancies, negative.Glucose,
                cmap="Reds", shade=True, shade_lowest=False)

#ax = sns.kdeplot(setosa.petal_width, setosa.petal_length,
#                  cmap="Reds", shade=True, shade_lowest=False)

#ax = sns.kdeplot(virginica.petal_width, virginica.petal_length,
#                  cmap="Blues", shade=True, shade_lowest=False)
#ax = sns.kdeplot(versicolor.sepal_width, versicolor.sepal_length,
#                  cmap="Greens", shade=True, shade_lowest=False)
