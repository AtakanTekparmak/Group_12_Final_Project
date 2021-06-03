import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join

# Plots the Daily Close Values of a Ticker
def plot_daily_close(ticker_name: str) -> None:

    file_path = "data" + os.path.sep + ticker_name
    # Reads Data and selects Date, Open, High, Low, Close and Volume columns for the Pandas dataframe
    dataset = pd.read_csv(file_path + ".csv", usecols = [0,1,2,3,4,6])

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


if __name__ == '__main__':
    #remove_null_values()
    ticker_name = input("Ticker name: ")
    plot_daily_close(ticker_name)
