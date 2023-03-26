# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from category_encoders import BinaryEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix

# Load the dataset
data = pd.read_csv('data\pay_gap_Europe.csv')

# Initialize the model
model = LinearRegression()

# Define the target variable
target_variable = "GDP"

# Identify the numerical and string columns
numerical_columns = data.select_dtypes(include=['number']).columns
string_columns = data.select_dtypes(include=['object']).columns

# Impute missing values for numerical columns with their mean
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = numerical_imputer.fit_transform(data[numerical_columns])

# Impute missing values for string columns with the most frequent value
string_imputer = SimpleImputer(strategy='most_frequent')
data[string_columns] = string_imputer.fit_transform(data[string_columns])

## Encode the string columns
# encoder = BinaryEncoder()
# data_encoded = encoder.fit_transform(data[string_columns])
data_encoded = pd.get_dummies(data, columns=['Country'])

# MinMax Scaling
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[numerical_columns])

# Combine the encoded and scaled columns
data_processed = pd.concat([data_encoded, pd.DataFrame(data_scaled, columns=numerical_columns)], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_processed.drop(target_variable, axis=1), data_processed[target_variable], test_size=0.2, random_state=42)

# Display the first few rows of the processed data
print(data_processed.head())

# Train the model on the training data
model.fit(X_train, y_train)

# Validate the model using cross validation
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross validation scores: ", scores)

# Make predictions using the model
y_pred = model.predict(X_test)

print("Predicted GDP values:", y_pred[:5])

# Calculate the metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# Convert y_test and y_pred to 1D arrays
y_test = np.array(y_test).flatten()
y_pred = np.array(y_pred).flatten()

# Create a scatter plot of predicted vs actual GDP values
# Add a line of best fit to the scatter plot
m, b = np.polyfit(y_test, y_pred, 1)
plt.scatter(y_test, y_pred)
plt.plot(y_test, m*y_test + b, color='red')
plt.xlabel("Actual GDP")
plt.ylabel("Predicted GDP")
plt.title("Predicted vs Actual GDP")
plt.show()