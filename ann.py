#Artificial Neural Network
"""
Created on Thu Oct 24 21:20:39 2019

@author: SHIVAM
"""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

#preprocessing the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder_1 = LabelEncoder()
X[:, 1] = encoder_1.fit_transform(X[:, 1])
encoder_2 = LabelEncoder()
X[:, 2] = encoder_2.fit_transform(X[:, 2])
ohe = OneHotEncoder(categorical_features = [1])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:]

#Splitting the dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Designing the ANN
#Importing the libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#Decalring the NN
classifier = Sequential()

#Initializing the NN using Dense()
#First hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

#Second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the NN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training the NN
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

#Predicting the compiler results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Plotting the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





