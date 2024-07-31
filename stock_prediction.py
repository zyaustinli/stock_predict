import yfinance
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib

def svr_predict():
    df = yfinance.download("GOOGL")
    #print(df.columns)

    df = df[['Adj Close']]

    #how many days into future predicted
    forecast_out = 30
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out)


    X = np.array(df.drop(['Prediction'], axis=1))
    X = X[:-forecast_out]


    y = np.array(df["Prediction"])
    y = y[:-forecast_out]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    """param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1e-3, 1e-4, 0.1, 0.01, 0.001],
        'epsilon': [0.1, 0.2, 0.5, 0.3]
    }

    svr_rbf = SVR(kernel = 'rbf')
    grid_search = GridSearchCV(svr_rbf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    svr_confidence = best_model.score(X_test, y_test)

    #best params: {'C': 1000, 'epsilon': 0.1, 'gamma': 0.0001}
    """

    svr_rbf = SVR(kernel='rbf', C=1000, gamma= 0.1, epsilon=0.1)
    svr_rbf.fit(X_train,y_train)
    svr_confidence = svr_rbf.score(X_test,y_test)
    #print(svr_confidence)

    x_forecast = np.array(df.drop(['Prediction'], axis=1))[-forecast_out:]

    svr_prediction = svr_rbf.predict(x_forecast)
    print(svr_prediction)



    '''
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_confidence = lr.score(X_test,y_test)
    print(lr_confidence)
    '''

svr_predict()