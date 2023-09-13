
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
#reference from kaggle
np.random.seed(42)
import pandas as pd
import numpy as np
#import time
import matplotlib.pyplot as plt
#from tree.base import DecisionTree
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from metrics import *
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE
np.random.seed(42)

# Read real-estate data set
# ...
# 
'''mpg = pd.read_csv("/Users/madhuriawachar/Downloads/auto-mpg.csv")


# import pandas as pd

import pandas as pd

# Load dataset
#mpg = pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")

# Inspect the dataset
#print(mpg.head(5))

# Print DataFrame information
mpg_info = mpg.info()
#print(mpg_info)


#Removing non numeric rows from the dataset

mpg=mpg[mpg.horsepower.apply(lambda x:x.isnumeric())]

#Looking at the datatype of horsepower column
mpg["horsepower"].dtype

#It is still object lst us convert it to float
mpg["horsepower"]=mpg["horsepower"].astype("float")

#Chehcking the datatype
assert mpg["horsepower"].dtype == "float"


# Droping the car name column
mpg = mpg.drop("car name", axis="columns")
#Looking at the new data frame
#print(mpg.head(5))

#look at the dataFrame information
#
mpg.info()


#train_test_split = int(0.7*len(X))



X_train,X_test,Y_train,Y_test = train_test_split(mpg[mpg.columns[1:-1]],mpg[mpg.columns[-1]],test_size=.3)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

mse_dt = MSE(Y_test,y_pred)

print('MSE:',mse_dt)
rmse_dt = mse_dt**(1/2)
print('RMSE:',rmse_dt)'''
#tree = DecisionTreeClassifier()
#tree.fit(X_train,Y_train)

# Building Decesion Tree based on my model
#criteria = 'information_gain'
#mytree = DecisionTree(criterion=criteria, max_depth=maxdepth) #Split based on Inf. Gain

data = pd.read_csv("/Users/madhuriawachar/Downloads/auto-mpg.csv")













#print('after transpose\n',data.head(5))
#Car name column can be dropped as this column cannot be fit into model
data.drop('car name', axis = 1, inplace = True)
"""print('data after car name \n',data.head(5))




print('pd.get_dummies \n',data.head(5))"""

#Replace '?' with NaN
data = data.replace('?', np.nan)
data[data.isnull().any(axis = 1)]
data["horsepower"]=data["horsepower"].astype("float64")
#instead of dropping the rows, lets replace the missing values with median value.
data = data.apply(lambda x: x.fillna(x.median()), axis = 0)

#horsepower column should have it's datatype changed from object to float
data["cylinders"] = data["cylinders"].astype("category")
data["origin"]=data["origin"].astype("category")
train_test_split = int(0.7*data.shape[0])
X = data.iloc[:train_test_split, 1:]
X_test = data.iloc[train_test_split:, 1:]
y = data.iloc[:train_test_split, 0]
y_test = data.iloc[train_test_split:, 0]
#X_train,X_test,Y_train,Y_test = train_test_split(data[data.columns[1:-1]],data[data.columns[-1]],test_size=.3)

# Create Decision Tree classifer object
clf = DecisionTreeRegressor()

# Train Decision Tree Classifer
clf = clf.fit(X,y)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

mse_dt = MSE(y_test,y_pred)
mae_=MAE(y_test,y_pred)
#print('MSE:',mse_dt)
rmse_dt = mse_dt**(1/2)
print('RMSE:',mse_dt**0.5)
print('MAE:',mae_)


#print(X)

#print(X)
#print(y)

maxdepth = 4

# Building Decesion Tree based on my model
criteria = 'information_gain'
mytree = DecisionTree(criterion=criteria, max_depth=maxdepth) #Split based on Inf. Gain
mytree.fit(X, y)
mytree.plot()
print("sklearn")
print('RMSE:',rmse_dt)
print('MAE:',mae_)

print("My Model")
y_hat = mytree.predict(X)
print("Train Scores:")
print('\tRMSE: ', rmse(y_hat, y))
print('\tMAE: ', mae(y_hat, y))

y_test_hat = mytree.predict(X_test)
print("Test Scores:")
print('\tRMSE: ', rmse(y_test_hat, y_test))
print('\tMAE: ', mae(y_test_hat, y_test))
