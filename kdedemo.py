# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:25:40 2017

@author: BALASUBRAMANIAM
"""

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

#plotting univaruate distributions both kde and histogram
x = np.random.normal(size=100)
print(x)
sns.distplot(x);#both hist and kde
sns.distplot(x, hist=False);#data samples will be visible
sns.kdeplot(x, shade=True);

iris = sns.load_dataset("iris")
sns.pairplot(iris);
g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);

