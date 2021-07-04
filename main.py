import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import math
from sklearn.metrics import mean_squared_error

from numpy.random import seed

# Global storage for losses
loss_over_tickers = [] # Average loss for each ticker
loss_over_time_tickers = [] # Losses for each epoch of each ticker


# Global hyperparameters for easy experimentation
# Changing the values of the dictionary changes the model archictecture/process
hyperparameters = {
    "LSTM_Unit_Number_1":  64, 
    "LSTM_Unit_Number_2":  32,
    "Dropout_Unit_Number": 0.2,
    "Batch_Size":          10, 
    "Epochs":              20, 
}


# Reads data from input generated csv file
def read_csv_file(ticker_name: str) -> None:

    file_path = "data" + os.path.sep + ticker_name
    
    # Reads Data and selects Date, Open, High, Low, Close and Volume columns for the Pandas dataframe
    dataset = pd.read_csv(file_path + ".csv", usecols = [0,1,2,3,4,6])
    return dataset


# Plots the Daily Close Values of a Ticker
def plot_daily_close(dataset, ticker_name):

    # Plots the Daily Prices of Selected Ticker
    plt.plot(dataset[['Close']], 'g', label = 'Closing price')
    plt.title(ticker_name + ' Data')
    plt.legend(loc = 'upper left')
    plt.show()


# Removes the null values in the datasets
def remove_null_values(data_path: str = "data"):

    files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    for file in files:

        # Remove Non Null
        file_path = data_path + os.path.sep + file
        dataset = pd.read_csv(file_path)
        dataset = dataset.dropna()
        dataset.to_csv(file_path)
        
        # Fix the Error in Column Names
        with open(file_path) as f:
            lines = f.readlines()

            lines[0] = "Date,Open,High,Low,Close,Adj Close,Volume\n"

            with open(file_path, "w") as f:
                f.writelines(lines)


# Splits dataframe into 80% training and 20% testing
def split_data(dataset):

    mean4 = dataset.mean(axis = 1)
    length = len(mean4)

    train_size = int(len(mean4) * 0.8)
    test_size = int(length - train_size)

    train = mean4[0:train_size]
    X = mean4[0:length - 1]
    y = mean4[1:length]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

    return X_train, X_test, y_train, y_test


# Standardizes Training and Testing using MinMaxScaler
def standardize_data(X_train, X_test, y_train, y_test):
    
    scaler = MinMaxScaler()

    X_train_size = len(X_train)
    X_test_size = len(X_test)

    X_train = np.reshape(X_train.values, (X_train_size, 1)) 
    X_train = scaler.fit_transform(X_train)
    
    X_test = np.reshape(X_test.values, (X_test_size, 1)) 
    X_test = scaler.fit_transform(X_test)

    y_train_size = len(y_train)
    y_test_size = len(y_test)

    y_train = np.reshape(y_train.values, (y_train_size, 1)) 
    y_train = scaler.fit_transform(y_train)
    
    y_test = np.reshape(y_test.values, (y_test_size, 1)) 
    y_test = scaler.fit_transform(y_test)

    return scaler, X_train, X_test, y_train, y_test


# Initializes LSTM model
def initialize_model():

    model = Sequential()
    model.add(LSTM(hyperparameters["LSTM_Unit_Number_1"], activation = 'relu', return_sequences = True, input_shape = (1, 1)))
    model.add(Dropout(hyperparameters["Dropout_Unit_Number"]))

    model.add(LSTM(hyperparameters["LSTM_Unit_Number_2"], activation = 'relu'))
    model.add(Dropout(hyperparameters["Dropout_Unit_Number"]))

    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'mse')

    return model


# Trains the model and predicts future values
def train_model(model, scaler, X_train, X_test, y_train, y_test):

    steps = 2
    feature_number = 1
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    history = model.fit(X_train, y_train, batch_size = hyperparameters["Batch_Size"], epochs = hyperparameters["Epochs"], verbose = 0)

    losses = history.history['loss']
    loss_over_tickers.append(sum(losses) / len(losses))
    loss_over_time_tickers.append(losses)

    predict_train = model.predict(X_train)
    predict_train = scaler.inverse_transform(predict_train)
    
    predict_test = model.predict(X_test)
    predict_test = scaler.inverse_transform(predict_test)

    y_train = scaler.inverse_transform(y_train)


    # Optional plot to compare training and testing values to the actual stock values

    #plt.plot(y_train, color = 'black', label = 'Real Stock Price')
    #plt.plot(predict_train, color = 'green', label = 'Predicted Stock Price - Train')
    #plt.plot([None for i in range(int(len(predict_train) * 8 / 10))] + [x for x in predict_test], color = 'orange', label = 'Predicted Stock Price - Test')

    #plt.title('Stock Price Prediction')
    #plt.xlabel('Time')
    #plt.ylabel('Stock Price')
    #plt.legend()
    #plt.show()


# Plots the loss over all tickers
def plot_loss_tickers():

    plt.plot(loss_over_tickers, color = 'red', label = 'Average Loss of Tickers')
    plt.title('Loss Over All 501 Tickers')
    plt.xlabel('Number of Tickers')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Processes a given ticker 
def process_ticker(ticker_name, model):

    dataset = read_csv_file(ticker_name)
    #plot_daily_close(dataset, ticker_name)
    X_train, X_test, y_train, y_test = split_data(dataset)
    scaler, X_train, X_test, y_train, y_test = standardize_data(X_train, X_test, y_train, y_test)
    train_model(model, scaler, X_train, X_test, y_train, y_test)


# Created by: Lauren Kersten (s3950905), Atakan Tekparmak (s4017765), Fransiskus Adrian Gunawan (s4024028), and Melisa Samancioglu (s4010418)


if __name__ == '__main__':

    #remove_null_values()
    seed(1)

    model = initialize_model()
    files = [f for f in listdir("data") if isfile(join("data", f))]

    i = 0 

    # Loops through all of the .csv files
    for file_name in files:
      process_ticker(file_name.rstrip(".csv"), model)
      if i % 10 == 0:
        print(str(i) + " out of " + str(len(files)))    # Prints progress of program
      i = i + 1

    plot_loss_tickers()

    print("Average loss = ")
    print(sum(loss_over_tickers)/len(loss_over_tickers))