# #feature_selection.py
# import pandas as pd

# # Load the preprocessed data that contains the calculated features
# data = pd.read_csv('preprocessed_data_with_features.csv')

# # Ensure no NaN values exist in selected features (due to rolling windows)
# data.dropna(subset=['SMA_20', 'RSI'], inplace=True)

# # Define the features to be used for model training
# features = ['SMA_20', 'RSI']

# # Select these features from the DataFrame
# X = data[features]

# # Save the selected features to a new CSV file for model training
# X.to_csv('features.csv', index=False)

# print(X.head())
# print("Selected features saved.")


# feature_selection.py
import pandas as pd

# Load the preprocessed data that contains the calculated features
data = pd.read_csv('preprocessed_data_with_features.csv')

# Ensure no NaN values exist in selected features (due to rolling windows)
data.dropna(subset=['SMA_20', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'Upper_BB', 'Lower_BB'], inplace=True)

# Define the features to be used for model training
features = ['SMA_20', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'Upper_BB', 'Lower_BB']

# Select these features from the DataFrame
X = data[features]

# Save the selected features to a new CSV file for model training
X.to_csv('features.csv', index=False)

print(X.head())
print("Selected features saved.")