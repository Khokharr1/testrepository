# #preprocessing.py
# import pandas as pd

# # Load the raw candlestick data
# data = pd.read_csv('candlestick_data.csv')

# # Ensure that the 'Close' column contains numeric data
# data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
# data.dropna(subset=['Close'], inplace=True)

# # Calculate the 5-period Simple Moving Average (SMA_5)
# data['SMA_5'] = data['Close'].rolling(window=5).mean()

# # Calculate the 20-period Simple Moving Average (SMA_20)
# data['SMA_20'] = data['Close'].rolling(window=20).mean()

# # Function to calculate the Relative Strength Index (RSI)
# def calculate_rsi(series, period=14):
#     delta = series.diff(1)
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)

#     avg_gain = gain.rolling(window=period).mean()
#     avg_loss = loss.rolling(window=period).mean()

#     rs = avg_gain / avg_loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi

# # Calculate the RSI with a 14-period window
# data['RSI'] = calculate_rsi(data['Close'], period=14)

# # Save the preprocessed data to a new CSV file
# data.to_csv('preprocessed_data_with_features.csv', index=False)
# print("Preprocessing completed, features calculated and data saved.")

# preprocessing.py
import pandas as pd

# Load the raw candlestick data
data = pd.read_csv('candlestick_data.csv')

# Ensure that the 'Close' column contains numeric data
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data.dropna(subset=['Close'], inplace=True)

# Calculate the 5-period Simple Moving Average (SMA_5)
data['SMA_5'] = data['Close'].rolling(window=5).mean()

# Calculate the 20-period Simple Moving Average (SMA_20)
data['SMA_20'] = data['Close'].rolling(window=20).mean()

# Calculate Exponential Moving Averages (EMA)
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

# Calculate MACD
data['MACD'] = data['EMA_12'] - data['EMA_26']

# Calculate Bollinger Bands
data['Upper_BB'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
data['Lower_BB'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)

# Function to calculate the Relative Strength Index (RSI)
def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate the RSI with a 14-period window
data['RSI'] = calculate_rsi(data['Close'], period=14)

# Drop rows with NaN values
data.dropna(inplace=True)

# Save the preprocessed data to a new CSV file
data.to_csv('preprocessed_data_with_features.csv', index=False)
print("Preprocessing completed, features calculated and data saved.")