# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:44:57 2017

@author: BALASUBRAMANIAM
"""

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np
ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268., 400., 754., 564., 138., 219., 869., 669.])

Z = hierarchy.linkage(ytdist, 'single')
dn = hierarchy.dendrogram(Z)
plt.show()
'''
hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y',orientation='top')
dn2 = hierarchy.dendrogram(Z, ax=axes[1], above_threshold_color='#bcbddc',orientation='right')
hierarchy.set_link_color_palette(None)  # reset to default after use
'''
