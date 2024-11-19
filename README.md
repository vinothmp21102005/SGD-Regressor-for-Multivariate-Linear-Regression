# SGD-Regressor-for-Multivariate-Linear-Regression
## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Use pandas, numpy, and sklearn modules for dataset handling, preprocessing, and modeling.
2. Fetch the California housing dataset and split features (x) and targets (y).
3. Scale x and y using StandardScaler for normalization.
4. Divide the dataset into training and testing sets using train_test_split.
5. Use MultiOutputRegressor with SGDRegressor to fit the training data.
6. Predict on the test set, inverse-transform predictions, and compute MSE.
## Program:
```
/*
Program to implement the multivariate linear regression model
for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: VINOTH M P
RegisterNumber:  212223240182
*/
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# load the california housing dataset
dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
df.head()

# use the first 3 features as input
x=df.drop(columns=['AveOccup','HousingPrice'])
x

y = df[['AveOccup','HousingPrice']]
y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scal_x =StandardScaler()
scal_y =StandardScaler()


x_train=scal_x.fit_transform(x_train)
x_test=scal_x.transform(x_test)

y_train=scal_y.fit_transform(y_train)
y_test=scal_y.transform(y_test)

#initilize SGDRegressor
sgd = SGDRegressor(max_iter=1000,tol=1e-3)

# use multioutputregressor to handle multuple output variables
mul_out=MultiOutputRegressor(sgd)

# train the model
mul_out.fit(x_train,y_train)

# predict on the test data
y_pred= mul_out.predict(x_test)
y_pred=scal_y.inverse_transform(y_pred)
y_test=scal_y.inverse_transform(y_test)

mse=mean_squared_error(y_test,y_pred)
print("mean_squared_error:",mse)
print("\nPredictions:\n",y_pred[:5])
```

## Output:

![image](https://github.com/user-attachments/assets/a40f716d-1340-4f64-bac3-6ad8906905ad)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
