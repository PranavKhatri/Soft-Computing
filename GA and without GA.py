# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:07:11 2019

@author: prana
"""

import numpy as np
import pandas as pd
import random as rd
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

# Loading the data, shuffling and preprocessing it

Data = pd.read_csv("C:/Users/prana/Downloads/file12.csv")

Data = Data.sample(frac=1)

X1 = pd.DataFrame(Data,columns=['X1','X2','X3','X4','X5','X6','X7','X8'])

Y = pd.DataFrame(Data,columns=['Y1']).values


Xbef = pd.get_dummies(X1,columns=['X6','X8'])


min_max_scalar = preprocessing.MinMaxScaler()

X = min_max_scalar.fit_transform(Xbef)

Cnt1 = len(X)
print()
print("# of Observations:",Cnt1)


kfold = 10

MLPClass = MLPRegressor()

Count1 = 1
Aa1 = 0

Cnt1 = len(X)

kf = KFold(n_splits=kfold)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    model1 = MLPClass
    model1.fit(X_train, Y_train)
    Pa_1=model1.predict(X_test)
    AC1=model1.score(X_test,Y_test)
    
    Aa1 += AC1
       
print()
print("R2 for MLP witout GA: %f" % (Aa1/kfold))




MLPClass = MLPRegressor(activation='relu',solver='adam',hidden_layer_sizes=(10,10,10),
                           learning_rate_init=0.025,momentum=0.8758)

Count1 = 1
Aa1 = 0

Cnt1 = len(X)



kf = KFold(n_splits=kfold)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    model1 = MLPClass
    model1.fit(X_train, Y_train)
    Pa_1=model1.predict(X_test)
    AC1=model1.score(X_test,Y_test)
    
    Aa1 += AC1
       
print()
print("R2 for MLP with GA: %f" % (Aa1/kfold))
