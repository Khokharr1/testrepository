# #model_training.py
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# import pickle

# # Load the original dataset
# features = pd.read_csv('features.csv')
# labeled_data = pd.read_csv('labeled_data.csv')

# # Get the target from the labeled data
# target = labeled_data['Target'].values

# # Make sure features and target have the same length
# min_length = min(len(features), len(target))
# features = features.iloc[:min_length]
# target = target[:min_length]

# # Split the data into training and testing sets (80/20 split)
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # Train a logistic regression model
# model = LogisticRegression(random_state=42)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Calculate accuracy and other metrics
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Example prediction on the latest data
# latest_data = features.iloc[-1:].values
# prediction = model.predict(latest_data)

# if prediction[0] == 1:
#     print("Price will go UP in the next period.")
# else:
#     print("Price will go DOWN in the next period.")

# # Save the trained model
# with open('price_prediction_model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# print("Model saved as 'price_prediction_model.pkl'")

# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the labeled data
data = pd.read_csv('labeled_data.csv')

# Define features and target
features = ['SMA_20', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'Upper_BB', 'Lower_BB']
X = data[features]
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the model and hyperparameter grid
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Best Hyperparameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
with open('price_prediction_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Model trained and saved.")