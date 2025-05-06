import numpy as np
import pandas as pd

df = pd.read_csv('BostonHousing.csv')
df
df.head(10)
df.isnull().sum()

df.dtypes

X = df[['crim','rm','lstat']]
Y = df['medv']

df.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

from sklearn.metrics import r2_score

y_train_predict = lin_model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
r2 = r2_score(y_train, y_train_predict)

print("Model Performance for training set")
print("-----------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

y_test_predict = lin_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
r2 = r2_score(y_test, y_test_predict)

print("Model Performance for testing set")
print("-----------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, y_test_predict)
mae = mean_absolute_error(y_test, y_test_predict)
print('Mean Squared Error : ',mse)
print('Mean Absolute Error : ',mae)

mse = mean_squared_error(y_train, y_train_predict)
mae = mean_absolute_error(y_train, y_train_predict)
print('Mean Squared Error : ',mse)
print('Mean Absolute Error : ',mae)




