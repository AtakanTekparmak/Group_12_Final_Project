import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler


# Reads data from input generated csv file
def read_csv_file(ticker_name: str) -> None:

	file_path = "data" + os.path.sep + ticker_name
    
    # Reads Data and selects Date, Open, High, Low, Close and Volume columns for the Pandas dataframe
	dataset = pd.read_csv(file_path + ".csv", usecols = [0,1,2,3,4,6])
	return dataset


# Plots the Daily Close Values of a Ticker
def plot_daily_close(database):

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
	test = mean4[train_size:len(mean4)]

	print('Length of Training Data: %d' % (len(train)))
	print(train)

	print('Length of Testing Data: %d' % (len(test)))
	print(test)

	plt.plot(train)
	plt.plot([None for i in train] + [x for x in test])
	plt.show()

	return train, test, train_size, test_size


# Standardizes Training and Testing using MinMaxScaler
def standardize_data(train, test, train_size, test_size):

	scaler = MinMaxScaler()

	train = np.reshape(train.values, (train_size, 1)) 
	train = scaler.fit_transform(train)
	print(train)

	test = np.reshape(test.values, (test_size, 1)) 
	test = scaler.fit_transform(test)
	print(test)


if __name__ == '__main__':
    #remove_null_values()
    ticker_name = input("Ticker name: ")
    dataset = read_csv_file(ticker_name)
    plot_daily_close(dataset)
    train, test, train_size, test_size = split_data(dataset)
    standardize_data(train, test, train_size, test_size)

