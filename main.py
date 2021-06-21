import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Reads data from input generated csv file
def read_csv_file(ticker_name: str) -> None:

    file_path = "data" + os.path.sep + ticker_name
    
    # Reads Data and selects Date, Open, High, Low, Close and Volume columns for the Pandas dataframe
    dataset = pd.read_csv(file_path + ".csv", usecols = [0,1,2,3,4,6])
    return dataset


# Plots the Daily Close Values of a Ticker
def plot_daily_close(dataset):

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


    print(train)
    print("X_train")
    print(X_train)
    print("X_test")
    print(X_test)
    print("y_train")
    print(y_train)
    print("y_test")
    print(y_test)


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

    print("Standardized X Train")
    print(X_train)
    print("Standardized X Test")
    print(X_test)

    y_train_size = len(y_train)
    y_test_size = len(y_test)

    y_train = np.reshape(y_train.values, (y_train_size, 1)) 
    y_train = scaler.fit_transform(y_train)
    
    y_test = np.reshape(y_test.values, (y_test_size, 1)) 
    y_test = scaler.fit_transform(y_test)

    print("Standardized y Train")
    print(y_train)
    print("Standardized y Test")
    print(y_test)

    return X_train, X_test, y_train, y_test

# Processes a given ticker 
def process_ticker(ticker_name):
    dataset = read_csv_file(ticker_name)
    plot_daily_close(dataset)
    X_train, X_test, y_train, y_test = split_data(dataset)
    X_train, X_test, y_train, y_test = standardize_data(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    #remove_null_values()
    ticker_name = input("Ticker name: ")
    process_ticker(ticker_name)