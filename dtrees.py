import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz 
from utils import load_water
from sklearn import tree

water = load_water()

X, y = water.data, water.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

plot_tree = tree.plot_tree(clf)
print(plot_tree)

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("water") 

dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=water.feature_names,  
                     class_names=water.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
print(graph)