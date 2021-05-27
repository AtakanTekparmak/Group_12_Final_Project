import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Reads Data and selects Open, High, Low, Close columns for the panda dataframe
dataset = pd.read_csv('AAPL.csv', usecols = [1,2,3,4])

# Vector of same length (goes one step at a time), used when plotting
days = np.arange(1, len(dataset) + 1, 1)

# Average of all of 4 columns (Open, High, Low, Close)
mean4 = dataset.mean(axis = 1) 

# New vector with the opening values of every day
opening = dataset[['Open']]

# New vector with closing values of every day
closing = dataset[['Close']]


# Plots the graphs 
plt.plot(days, mean4, 'b', label = 'OHLC mean')
plt.plot(days, opening, 'y', label = 'Opening price')
plt.plot(days, closing, 'g', label = 'Closing price')
plt.title('Stock Data')
plt.legend(loc = 'upper left')
plt.show()

