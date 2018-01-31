# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 00:42:01 2017

@author: BALASUBRAMANIAM
"""
import sys
sys.path.append('C:/Program Files (x86)/Graphviz2.38/bin')

import pydotplus
import graphviz as gv
from sklearn.datasets import load_iris
from sklearn import tree
import collections
 
# Data Collection 'height', 'hair length', 'voice pitch' 
X = [ [180, 15,0],     
      [177, 42,0],
      [136, 35,1],
      [174, 65,0],
      [141, 28,1]]
 
Y = ['man', 'woman', 'woman', 'man', 'woman']    
 
data_feature_names = [ 'height', 'hair length', 'voice pitch' ]

# Training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
import sys
sys.path.append('C:/Program Files (x86)/Graphviz2.38/bin')
# Visualize data

dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph1 = pydotplus.graph_from_dot_data(dot_data)
 
colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)
 
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
 
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
 
print('graph ready')    
#graph.write_png('file.png')

#print(graph.source)
#graph1.write_pdf('iris.pdf')
#graph.write_png('test.png')


g1 = gv.Graph(format='svg')
g1.node('A')
g1.node('B')
g1.edge('A', 'B')
print(g1.source)






import pandas as pd
import numpy as np
from sklearn import tree


# creating dataset for modeling Apple / Orange classification
fruit_data_set = pd.DataFrame()
fruit_data_set["fruit"] = np.array([1, 1, 1, 1, 1,      # 1 for apple
                                    0, 0, 0, 0, 0])     # 0 for orange
fruit_data_set["weight"] = np.array([170, 175, 180, 178, 182,
                                     130, 120, 130, 138, 145])
fruit_data_set["smooth"] = np.array([9, 10, 8, 8, 7,
                                     3, 4, 2, 5, 6])

fruit_classifier = tree.DecisionTreeClassifier()
fruit_classifier.fit(fruit_data_set[["weight", "smooth"]], fruit_data_set["fruit"])

print (">>>>> Trained fruit_classifier <<<<<")
print (fruit_classifier)

# fruit data set 1st observation
test_features_1 = [[fruit_data_set["weight"][0], fruit_data_set["smooth"][0]]]
test_features_1_fruit = fruit_classifier.predict(test_features_1)
print ("Actual fruit type: {act_fruit} , Fruit classifier predicted: {predicted_fruit}".format(
    act_fruit=fruit_data_set["fruit"][0], predicted_fruit=test_features_1_fruit))

# fruit data set 3rd observation
test_features_3 = [[fruit_data_set["weight"][2], fruit_data_set["smooth"][2]]]
test_features_3_fruit = fruit_classifier.predict(test_features_3)
print ("Actual fruit type: {act_fruit} , Fruit classifier predicted: {predicted_fruit}".format(
    act_fruit=fruit_data_set["fruit"][2], predicted_fruit=test_features_3_fruit))

# fruit data set 8th observation
test_features_8 = [[fruit_data_set["weight"][7], fruit_data_set["smooth"][7]]]
test_features_8_fruit = fruit_classifier.predict(test_features_8)
print ("Actual fruit type: {act_fruit} , Fruit classifier predicted: {predicted_fruit}".format(
    act_fruit=fruit_data_set["fruit"][7], predicted_fruit=test_features_8_fruit))


with open("fruit_classifier.txt", "w") as f:
    f = tree.export_graphviz(fruit_classifier, out_file=f)

# converting into the pdf file
with open("fruit_classifier.dot", "w") as f:
    f = tree.export_graphviz(fruit_classifier, out_file=f)






from sklearn.datasets import load_iris
from sklearn import tree
clf = tree.DecisionTreeClassifier()
iris = load_iris()
clf = clf.fit(iris.data, iris.target)
tree.export_graphviz(clf,  out_file='tree.dot')   




from sklearn import tree
from sklearn.datasets import load_iris
from IPython.display import Image
import io

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# Let's give dot_data some space so it will not feel nervous any more
dot_data = io.StringIO()
tree.export_graphviz(clf, out_file=dot_data)
import pydotplus

graph = pydotplus.graphviz.graph_from_dot_data(dot_data.getvalue())
# make sure you have graphviz installed and set in path
Image(graph.create_png())

#graph.write_png('file.png')


  
#graph.write_png('tree.png')