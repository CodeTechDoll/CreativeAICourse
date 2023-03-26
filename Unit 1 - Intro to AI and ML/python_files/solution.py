# Import necessary libraries
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('data\pay_gap_Europe.csv')

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Display the first few rows of the dataset
print(data.head())