# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:44:30 2019

@author: SHIVAM
"""
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Scaling the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

#Training the SOM
#Sigma is the radius of the circle around the winning node
#Concept is based on finding the Mean Internode Distance 
#Higher the MID farther would be the winning node from the neigbors and hence is an outlier
#We will take the winning nodes with the highest MID because they are the frauds

from minisom import MiniSom
som = MiniSom(x = 10, y =10, input_len = 15, learning_rate = 0.5, sigma = 1.0)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)