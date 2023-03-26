# Standardization

Standardization is a technique used in data preprocessing to transform the data to have a mean of 0 and a standard deviation of 1. It is also known as "Z-score normalization" or "Z-score scaling".

Standardization is useful when the features in the data have different scales, such as in the case of features that have different units of measurement or different ranges of values. By standardizing the data, we can make the features more comparable and reduce the impact of the differences in scale.

The standardization formula for a single feature x is:

`z = (x - mean) / std` where z is the standardized value, mean is the mean of the feature values, and std is the standard deviation of the feature values.

To standardize a dataset with multiple features, we can apply the formula to each feature independently.

## Here's a real-world example to illustrate standardization:

Suppose we have a dataset of student exam scores for a math class, with two features: "score on exam 1" (ranging from 0 to 100) and "time spent studying" (measured in hours). We want to predict the score on exam 2 based on these features.

The "score on exam 1" feature has a range of values from 0 to 100, while the "time spent studying" feature has a range of values from 0 to several hundred hours. Because the two features have different scales, it is difficult to compare their relative importance in predicting the score on exam 2.

To standardize the data, we can calculate the mean and standard deviation of each feature, and then apply the standardization formula to each value. For example, if the mean score on exam 1 is 70 and the standard deviation is 10, a score of 80 would be standardized as:

`z = (80 - 70) / 10 = 1`

Similarly, if the mean time spent studying is 50 hours and the standard deviation is 20 hours, a value of 70 hours would be standardized as:

`z = (70 - 50) / 20 = 1`

After standardizing the data, the two features will have a mean of 0 and a standard deviation of 1, making them more comparable and reducing the impact of the differences in scale. We can now use the standardized features to build a predictive model for the score on exam 2.

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

# create a sample dataframe
df = pd.DataFrame({
    'score on exam 1': [80, 70, 90, 85, 95],
    'time spent studying': [70, 50, 80, 60, 90]
})

# create a StandardScaler object
scaler = StandardScaler()

# standardize the dataframe
df_std = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(df_std)
```

In this example, we create a sample dataframe df with two features: "score on exam 1" and "time spent studying". We then create a StandardScaler object and use its fit_transform method to standardize the dataframe. The resulting df_std dataframe has each feature standardized to have a mean of 0 and a standard deviation of 1.

Note that the resulting df_std dataframe has the same number of rows and columns as the original df dataframe, but with the feature values standardized. You can use df_std as input to machine learning models or further data analysis.