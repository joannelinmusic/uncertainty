import time
from data_folder_path import folder_path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score

start_time = time.time()


full_test_data = pd.read_csv(folder_path + 'full_test_data.csv')
full_train_data = pd.read_csv(folder_path + 'full_train_data.csv')
full_validation_data = pd.read_csv(folder_path + 'full_validation_data.csv')
high_test_data = pd.read_csv(folder_path + 'high_test_data.csv')
high_train_data = pd.read_csv(folder_path + 'high_train_data.csv')
high_validation_data = pd.read_csv(folder_path + 'high_validation_data.csv')
low_test_data = pd.read_csv(folder_path + 'low_test_data.csv')
low_train_data = pd.read_csv(folder_path + 'low_train_data.csv')
low_validation_data = pd.read_csv(folder_path + 'low_validation_data.csv')
no_test_data = pd.read_csv(folder_path + 'no_test_data.csv')
no_train_data = pd.read_csv(folder_path + 'no_train_data.csv')
no_validation_data = pd.read_csv(folder_path + 'no_validation_data.csv')


X_train_flat = np.array([np.array(image.replace('[', '').replace(']', '').split(','), dtype=int).flatten() for image in full_train_data['flattened_image']])
X_test_flat = np.array([np.array(image.replace('[', '').replace(']', '').split(','), dtype=int).flatten() for image in full_test_data['flattened_image']])

y_train = full_train_data['Label'].values
y_test = full_test_data['Label'].values

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train_flat, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test_flat)

# Calculate the Mean Squared Error (MSE) to evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r_squared = model.score(X_test_flat, y_test)
print("R²:", r_squared)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)