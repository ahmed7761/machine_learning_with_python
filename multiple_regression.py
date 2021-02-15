# Importing the essential libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#Importing the dataset 
dataset = pd.read_csv('50_Startups.csv') 
X = dataset.iloc[:, [0,1,2,3]].values 
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# remove categorical_features, it works 100% perfectly
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap 
X = X[:, 1:]

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression() 
regressor.fit(X_train, y_train)

#Predicting the Test set result 
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error 
print("The Mean Squared Error is- {}".format(mean_squared_error(y_test, y_pred))) 