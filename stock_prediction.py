import yfinance
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def data_pre(stock_name, forecast_out):
    df = yfinance.download(stock_name)
    #print(df)

    df1 = df[['Adj Close']]

    #how many days into future predicted
    df1['Prediction'] = df1[['Adj Close']].shift(-forecast_out)


    X = np.array(df1.drop(['Prediction'], axis=1))
    X = X[:-forecast_out]


    y = np.array(df1["Prediction"])
    y = y[:-forecast_out]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    svr_predict(X_train, X_test, y_train, y_test, df, df1, forecast_out)

def svr_predict(X_train, X_test, y_train, y_test, df, df1, forecast_out):
    


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

    x_forecast = np.array(df1.drop(['Prediction'], axis=1))[-forecast_out:]

    svr_prediction = svr_rbf.predict(x_forecast)
    print(svr_prediction)


    plot_predict(df,svr_prediction)



    '''
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_confidence = lr.score(X_test,y_test)
    print(lr_confidence)
    '''

def plot_predict(df_original, prediction):
    #plot takes two np arrays
    og_date = np.array(df_original.iloc[-30:].index)
    og_date_array = []
    for i in range(len(og_date)):
        og_date_array.append(str(og_date[i]).split('T')[0])

    #print(og_date_array)

    num_days = len(prediction)
    new_date = pd.date_range(start = pd.to_datetime(og_date_array[-1]) + pd.Timedelta(days=1), periods = num_days, freq='B').strftime('%Y-%m-%d').tolist()


    all_dates = np.array(og_date_array)
    all_dates = np.append(all_dates, new_date)
    #print(all_dates)

    all_prices = np.array(df_original.iloc[-30:]["Adj Close"])
    all_prices = np.append(all_prices, prediction)
    #print(all_prices)

    plt.plot(og_date_array, df_original.iloc[-30:]["Adj Close"], color="green")
    plt.plot(new_date, prediction, color="red", label="Prediction")

    plt.plot([og_date_array[-1], new_date[0]], 
             [df_original.iloc[-1]["Adj Close"], prediction[0]], 
             color="red")
    

    #plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    
    dates_plot = []
    for i in range(0,len(og_date_array), 7):
        dates_plot.append(og_date_array[i])
    for i in range(0, len(new_date), 7):
        dates_plot.append(new_date[i])
    #print(dates_plot)
    plt.xticks(dates_plot)
    plt.xticks(rotation=270)


    plt.show()
    return

data_pre("GOOGL", 1)


