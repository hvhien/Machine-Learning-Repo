#Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Encoding the independent variable 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoid the dummy variable trap
X = X[:,1:]


# Split the dataset into the training set and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

X_save = X_train[:,2:3]

#Fit to the training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_save,y_train)

#Predict


# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
## Step 1
X_opt = X[:, [0,1,2,3,4,5]]
## Step 2
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
## Step 3
print(reg_OLS.summary())
## Step 1
X_opt = X[:, [0,1,3,4,5]]
## Step 2
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
## Step 3
print(reg_OLS.summary())
## Step 1
X_opt = X[:, [0,3,4,5]]
## Step 2
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
## Step 3
print(reg_OLS.summary())
## Step 1
X_opt = X[:, [0,3,5]]
## Step 2
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
## Step 3
print(reg_OLS.summary())
X_opt = X[:, [0,3]]
## Step 2
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
## Step 3
print(reg_OLS.summary())




