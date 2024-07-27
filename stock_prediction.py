import yfinance
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV


df = yfinance.download("GOOGL")
#print(df.columns)

df = df[['Adj Close']]

#how many days into future predicted
forecast_out = 1
df['Prediction'] = df[['Adj Close']].shift(-forecast_out)


X = np.array(df.drop(['Prediction'], axis=1))
X = X[:-forecast_out]


y = np.array(df["Prediction"])
y = y[:-forecast_out]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma=0.1)
svr_rbf.fit(X_train, y_train)
svr_confindence = svr_rbf.score(X_test,y_test)
print(svr_confindence)



'''
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_confidence = lr.score(X_test,y_test)
print(lr_confidence)
'''