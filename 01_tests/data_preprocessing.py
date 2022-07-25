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