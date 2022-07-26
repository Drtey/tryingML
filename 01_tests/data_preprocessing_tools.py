#Importing libraries

import numpy as np
import matplotlib as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #independent variable 
y = dataset.iloc[:, -1].values #dependent varible

print(X)

print(y)

#Missing data "NAN"
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
imputer.fit(X[:, 1:3])  #getting average row to the nan fields
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)

#OneHotEncoding ------ Encoding categorical data (transform strings to 0 and 1)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X)) #Fit and transform in one step

print(X)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1) 
#80% are going to the training set and the 20% are going to the test set

print(X_train)

print(X_test)

print(y_train)

print(y_test)

#Feature Scaling
#----Standardisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#----------ignore dummy variables to improve the perfomance of our model
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)

print(X_test)

