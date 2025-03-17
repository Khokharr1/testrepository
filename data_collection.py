#data_collection.py
import yfinance as yf
import pandas as pd

# Define the ticker and download the data
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2025-01-01', interval='1d')

# Select only the necessary columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.to_csv('candlestick_data.csv', index=True)

print("Data collected and saved.")
