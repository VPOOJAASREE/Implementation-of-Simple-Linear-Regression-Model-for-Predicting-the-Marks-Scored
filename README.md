# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: V. POOJAA SREE
RegisterNumber: 212223040147

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test

```




```
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
```
```
df.head()
'''




![1](https://github.com/user-attachments/assets/55753030-0631-4c72-90ba-96b058ef06df)




```
def.tail()
```


![2](https://github.com/user-attachments/assets/813ee0a3-7d11-48bc-8c12-209646f2243e)



```
x = df.iloc[:,:-1].values
x
```


![3](https://github.com/user-attachments/assets/d7f914d3-2dfc-4c97-9ba7-6b1f3128161d)


```
y = df.iloc[:,1].values
y
```


![4](https://github.com/user-attachments/assets/9f2f9f06-4cff-4cf2-a6c7-3f7a259885fb)



```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
```


![5](https://github.com/user-attachments/assets/d44dffc1-9cbf-470d-9f7c-0991707f07f3)


```
y_test
```



![6](https://github.com/user-attachments/assets/df15487d-4bf6-44be-b1bb-d94c11c1d10d)

![7](https://github.com/user-attachments/assets/e55c6f3c-3634-42c0-a66d-8674263dadec)



```
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```



![8](https://github.com/user-attachments/assets/c6b66214-e7d9-4242-a6be-be5786b0b6ed)



```
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```



![9](https://github.com/user-attachments/assets/441efb7e-88aa-4203-916c-5c9246a42eb3)

![10](https://github.com/user-attachments/assets/1d81fcee-0e30-4ffc-b1b4-2a0e9188069f)

![11](https://github.com/user-attachments/assets/0042ed0f-1e9e-4047-b1de-8b8108c9e19f)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
