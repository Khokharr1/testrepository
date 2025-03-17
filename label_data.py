#label_data.py
import pandas as pd
import numpy as np

# Load the preprocessed data
data = pd.read_csv('preprocessed_data_with_features.csv', index_col=0)

# Create binary labels: 1 if the price goes up, 0 if it goes down
data['Price_Change'] = data['Close'].shift(-1) - data['Close']
data['Target'] = np.where(data['Price_Change'] > 0, 1, 0)

# Drop the last row since we can't predict the future for the last candle
data = data[:-1]

# Save labeled data
data.to_csv('labeled_data.csv', index=True)

# Save target labels separately in 'target.csv'
target = data[['Target']]
target.to_csv('target.csv', index=True)

print("Labels created and data saved to 'labeled_data.csv' and 'target.csv'.")
