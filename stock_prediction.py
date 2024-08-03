import yfinance
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
import keras
from keras import Sequential
from keras import layers
from copy import deepcopy



def data_pre(stock_name, forecast_out, model_num):

    #1: linear regression #2: SVR

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

    if model_num == 1:
        linear_regression_predict(X_train, X_test, y_train, y_test, df, df1, forecast_out)
    elif model_num == 2:
        svr_predict(X_train, X_test, y_train, y_test, df, df1, forecast_out)
    elif model_num ==3:
        lstm_predict(df,forecast_out)

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
    #svr_confidence = svr_rbf.score(X_test,y_test)
    #print(svr_confidence)

    x_forecast = np.array(df1.drop(['Prediction'], axis=1))[-forecast_out:]

    svr_prediction = svr_rbf.predict(x_forecast)
    print(svr_prediction)


    plot_predict(df,svr_prediction)

def linear_regression_predict(X_train, X_test, y_train, y_test, df, df1, forecast_out):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    #lr_confidence = lr.score(X_test,y_test)
    x_forecast = np.array(df1.drop(['Prediction'], axis=1))[-forecast_out:]

    lr_prediction = lr.predict(x_forecast)
    print(lr_prediction)
    plot_predict(df,lr_prediction)

def lstm_predict(df, forecast_out):
    #use window size 3

    adj_close = df['Adj Close']
    dates = df.index
    date_strings = dates.strftime('%Y-%m-%d').tolist()

    #print(dates)
    window_df = pd.DataFrame(date_strings[3:], columns= ['Target Date'])

    #first date would be 3 after earliest date
    target_3 = adj_close[:-3].to_list()
    target_2 = adj_close[1:-2].to_list()
    target_1 = adj_close[2:-1].to_list()

    target = adj_close[3:].to_list()
    window_df['Target-3'] = target_3
    window_df['Target-2'] = target_2
    window_df['Target-1'] = target_1
    window_df['Target'] = target
    
    #target-3, -2, -1 are the input, and target is the output. model will predict based on previous 3 days

    df_to_np = window_df.to_numpy()
    dates  = df_to_np[:, 0]
    mid_matrix = df_to_np[:, 1:-1]
    X = mid_matrix.reshape((len(dates), mid_matrix.shape[1], 1)).astype(np.float32)
    y = df_to_np[:, -1].astype(np.float32)

    #sliding window with size 30, split into training: 24, test: 3, validation: 3

    X_train, y_train, X_test, y_test, X_val, y_val = [],[],[],[],[],[]

    #make copy of X and y
    X_copy = np.copy(X)
    y_copy = np.copy(y)

    iters = int(len(dates)/30)
    #print(iters)
    for i in range(iters):
        X_train.extend(X_copy[:24]) 
        y_train.extend(y_copy[:24])
        X_val.extend(X_copy[24:27])
        y_val.extend(y_copy[24:27])
        X_test.extend( X_copy[27:30])
        y_test.extend(y_copy[27:30])
        X_copy = X_copy[30:]
        y_copy = y_copy[30:]
    
    #add remaining data to train set
    X_train.extend(X_copy[:])
    y_train.extend(y_copy[:])
    #print(X_train[0:2])
    #print('\n\n\n')
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])
    model.compile(loss='mse', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
    
    last_window = deepcopy(X[-1])
    y_copy2 = deepcopy(y)
    next_value1 = model.predict(np.array([last_window])).flatten()

    predictions = []
    for _ in range(forecast_out):
        y_copy2 = np.append(y_copy2, next_value1[0]) #add to y for next prediction based on last 3 y values
        predictions.extend(next_value1) #add predicted value to predictions list
        next_window = np.array(y_copy2[-3:]).reshape((1,3,1))
        print(next_window)
        next_value1 = model.predict(next_window).flatten()

        

    print(predictions)
    plot_predict(df, predictions)

    '''X_train = np.concatenate([np.array(x) for x in X_train])
    y_train = np.concatenate([np.array(y) for y in y_train])
    X_val = np.concatenate([np.array(x) for x in X_val])
    y_val = np.concatenate([np.array(y) for y in y_val])
    X_test = np.concatenate([np.array(x) for x in X_test])
    y_test = np.concatenate([np.array(y) for y in y_test])
    print(X_train[0:2])'''

    '''#test with split data
    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    X_train, y_train = X[:q_80], y[:q_80]

    X_val, y_val = X[q_80:q_90], y[q_80:q_90]
    X_test, y_test = X[q_90:], y[q_90:]
'''

     #test with random
    #it is bad idea to do random selection of data in train and test when it comes to time series!!!! 

    ''' X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42) 
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42) 
    

    model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])
    model.compile(loss='mse', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
    
    last_window = deepcopy(X[-1])
    next_value2 = model.predict(np.array([last_window])).flatten()

    '''



    #print(next_value1)

    

    return


def plot_predict(df_original, prediction):
    #plot takes two np arrays
    og_date = np.array(df_original.iloc[-30:].index)
    og_date_array = []
    for i in range(len(og_date)):
        og_date_array.append(str(og_date[i]).split('T')[0])

    #print(og_date_array)

    num_days = len(prediction)
    new_date = pd.date_range(start = pd.to_datetime(og_date_array[-1]) + pd.Timedelta(days=1), periods = num_days, freq='B').strftime('%Y-%m-%d').tolist()


    '''all_dates = np.array(og_date_array)
    all_dates = np.append(all_dates, new_date)
    #print(all_dates)

    all_prices = np.array(df_original.iloc[-30:]["Adj Close"])
    all_prices = np.append(all_prices, prediction)
    #print(all_prices)'''

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

data_pre("MSFT", 10,3 )


