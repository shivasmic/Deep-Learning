# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:09:02 2019

@author: SHIVAM
"""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset.iloc[:, 1:2].values

#Scaling the data using Normalization technique
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Creating the train data using 60 timesteps
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
#Converting the data into an array
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping the training data since it needs three dimensional data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#Building the RNN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#We will create a regressor model as we are predicting a continuous value i.e price
regressor = Sequential()

#First LSTM layer and Dropout regularization
regressor.add(LSTM(units = 80, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Second layer and Dropout regularization
regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))

#Third layer and Dropout regularization
regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))

#Fourth layer and Dropout regularization
regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))

#Fifth layer and Dropout regularization
regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))

#Sixth layer and Dropout regularization
regressor.add(LSTM(units = 80))
regressor.add(Dropout(0.2))

#Output layer
regressor.add(Dense(units = 1))

#Compiling the regressor model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Training the RNN with test data
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#Predicting the prices for 2017
dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#Visulailsing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Prices')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Prices')
plt.title('Price Comparison')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
    

