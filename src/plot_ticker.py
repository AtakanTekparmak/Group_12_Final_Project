import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def plot_daily_close(ticker_name, cwd):

    # Reads Data and selects Date, Open, High, Low, Close and Volume columns for the Pandas dataframe
    dataset = pd.read_csv(r'%s' %cwd + os.path.sep + ticker_name + ".csv", usecols = [0,1,2,3,4,6])

    # Plots the Daily Prices of Selected Ticker
    plt.plot(dataset[['Close']], 'g', label = 'Closing price')
    plt.title(ticker_name + ' Data')
    plt.legend(loc = 'upper left')
    plt.show()


if __name__=='__main__':
	ticker_name = input("Ticker name: ")
	cwd = os.getcwd()
	cwd = os.path.split(cwd)
	cwd = os.path.join(cwd[0], "data")

	plot_daily_close(ticker_name, cwd)

