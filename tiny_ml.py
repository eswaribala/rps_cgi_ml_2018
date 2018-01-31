# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 20:24:40 2017

@author: BALASUBRAMANIAM
"""
import os
import scipy as sp

from utils import DATA_DIR, CHART_DIR

def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)
'''
numpy.genfromtxt(fname, dtype=<type 'float'>, comments='#', delimiter=None, skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=None, replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None)[source]
Load data from a text file, with missing values handled as specified.
'''

data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
print(data[:10])
print(data.shape)
#Dimesion separation
x = data[:,0]
print(x)
y = data[:,1]
print(y)
#data cleaning 
Nan=sp.sum(sp.isnan(y))
print(Nan) #missing 8 out of 743 data

x = x[~sp.isnan(y)]
print(x)
y = y[~sp.isnan(y)]
print(y)

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
'''
print('Week Information....')
for w in range(10):
    print(w*7*24)
    print('Week %d' %w)


print('Week Information....')
'''
plt.xticks([w*7*24 for w in range(10)],
['week %i'%w for w in range(10)])

'''
Axes.autoscale(enable=True, axis='both', tight=None)
Autoscale the axis view to the data (toggle).

Convenience method for simple axis view autoscaling. It turns autoscaling on or off, and then, if autoscaling for either axis is on, it performs the autoscaling on the specified axis or axes.

enable: [True | False | None]
True (default) turns autoscaling on, False turns it off. None leaves the autoscaling state unchanged.
axis: [‘x’ | ‘y’ | ‘both’]
which axis to operate on; default is ‘both’
tight: [True | False | None]
If True, set view limits to data limits; if False, let the locator and margins expand the view limits; if None, use tight scaling if the only artist is an image, otherwise treat tight as False. The tight setting is retained for future autoscaling until it is explicitly changed.
Returns None.
'''
plt.autoscale(tight=True)
plt.grid()
plt.savefig(os.path.join(CHART_DIR, "2017_09_23.png"))
plt.show()

#simple straight line
'''
p = polyfit(x,y,n)

Given
data x and y and the desired order of the 
polynomial (straight line has order 1),
it finds the model function that minimizes 
the error function defined earlier.
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
The polyfit() function returns the parameters of the fitted model function,
fp1; and by setting full to True, we also get additional background information
on the fitting process.
'''
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 10, full=True)


print("Model parameters: %s" % fp1)

print(residuals)

'''
This means that the best straight line fit is 
the following function:
f(x) = 2.59619213 * x + 989.02487106.
'''
f1 = sp.poly1d(fp1)

print(error(f1, x, y))
fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting
plt.plot(fx, f1(fx),'C8',linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left")



