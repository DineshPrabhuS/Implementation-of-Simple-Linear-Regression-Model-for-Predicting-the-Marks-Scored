# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the needed packages.

2.Assigning hours to x and scores to y.

3.Plot the scatter plot.

4.Use MSE, RMSE, MAE formula to find the values. 

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: DINESH PRABHU S

RegisterNumber: 212224040077
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df

df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

##plotting for training data
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="orange")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

##plotting for test data
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,y_pred,color="yellow")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:

![image](https://github.com/user-attachments/assets/749523e1-67a1-4cff-99c2-4fb9a12afa57)

![image](https://github.com/user-attachments/assets/9156341d-b86a-4ade-a2a0-20b222bc69c9)

![image](https://github.com/user-attachments/assets/d60eb89d-cacd-44d8-9aa3-27d56fb11cf9)

![image](https://github.com/user-attachments/assets/5534becb-8c79-4184-9e1d-0d05aaba6418)

![image](https://github.com/user-attachments/assets/b6d872a2-9dea-44df-875e-d4e0517c40fd)

![image](https://github.com/user-attachments/assets/81126166-87dd-4382-be2c-1f60a3fcd657)

![image](https://github.com/user-attachments/assets/436b883b-ecfe-414b-92dc-70b96e6f61d0)

![image](https://github.com/user-attachments/assets/25035103-b7d9-417e-bdb2-ad3cc22572a4)

![image](https://github.com/user-attachments/assets/a215ae5d-1d70-4458-89e5-d83ce0759d4c)

![image](https://github.com/user-attachments/assets/c2e64476-b1f7-48cd-9bde-a00435a92158)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
