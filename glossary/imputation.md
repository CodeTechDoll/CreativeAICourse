# Imputation

Imputation is a technique for handling missing data in a dataset by filling in the missing values with estimated values. The goal of imputation is to make the dataset complete so that it can be used for analysis. Imputation is a common practice in data science because it is rare to have a dataset that does not have any missing data.

There are many imputation strategies available, and the choice of strategy depends on the nature of the missing data and the goal of the analysis. In general, imputation strategies can be divided into three categories: mean imputation, median imputation, and mode imputation.

## Strategies

### Mean Imputation

Mean imputation is a method of imputing missing values in a dataset by replacing the missing values with the mean value of the non-missing values in the same column. This method is widely used for continuous variables, and it is commonly used when the missing data is assumed to be "Missing At Random" (MAR). Mean imputation is a relatively simple and fast method for handling missing data.
Procedure

The steps involved in mean imputation are:

- Identify the variables with missing values.
- Calculate the mean value of the non-missing values in the same column.
- Replace the missing values with the mean value.

Here's an example of how to perform mean imputation with scikit-learn:

```python
from sklearn.impute import SimpleImputer

# create imputer object with the mean strategy
imputer = SimpleImputer(strategy='mean')

# impute missing values in X
X = imputer.fit_transform(X)
```

In this example, X is the data matrix containing missing values, and SimpleImputer is an object that performs imputation. The strategy parameter is set to 'mean', which means that mean value will be used for imputing missing values in continuous variables.

#### Advantages

- It is simple to use and understand.
- It works well when the missing values are MAR.
- It preserves the mean and variance of the variable, which can be important for downstream analysis.

#### Disadvantages

- It assumes that the missing values are MAR, which may not always be the case.
- It can produce biased estimates if the missing values are not MAR.
- It may reduce the variance of the variable, which can be problematic for some downstream analysis.
- It can produce imputed values that are not realistic or plausible.
  
### Median Imputation

Median imputation is a method of imputing missing values in a dataset by replacing the missing values with the median value of the non-missing values in the same column. This method is commonly used for continuous variables, and it is suitable when the missing data is MAR (Missing At Random).

```python

from sklearn.impute import SimpleImputer

# create imputer object with the median strategy
imputer = SimpleImputer(strategy='median')

# impute missing values in X
X = imputer.fit_transform(X)
```

#### Advantages

- It is simple to use and understand.
- It works well for continuous variables.
- It is robust to outliers.
- It does not assume any underlying distribution, making it suitable for non-parametric data.

#### Disadvantages

- It may lead to biased estimates if the missing values are not MAR.
- It can produce imputed values that are not realistic or plausible.
- It may not be suitable for variables with extreme values that are not normally distributed.
- It does not take into account the correlation between variables.

### Mode Imputation

Mode imputation is a strategy that involves replacing missing values with the most frequent value of the non-missing values in the same column. Mode imputation is commonly used for categorical variables or when the data is non-parametric.

For datasets that include missing fields for categorical variables, such as a country, the most appropriate imputation method is mode imputation. Mode imputation replaces missing values with the most frequent value (i.e., the mode) of the non-missing values in the same column.

This method is suitable for categorical variables because the mode represents the most commonly occurring value in the variable, which is likely to be the best estimate for the missing value. Mode imputation is also robust to outliers and does not assume any underlying distribution, making it suitable for non-parametric data.

Here's an example of how to use mode imputation with scikit-learn:

```python
from sklearn.impute import SimpleImputer

# create imputer object with the mode strategy
imputer = SimpleImputer(strategy='most_frequent')

# impute missing values in X
X = imputer.fit_transform(X)
```

#### Advantages

- It is simple to use and understand.
- It works well for categorical variables.
- It is robust to outliers.
- It does not assume any underlying distribution, making it suitable for non-parametric data.

#### Disadvantages

- It may lead to biased estimates if the most frequent value is not representative of the true underlying distribution.
- It can produce imputed values that are not realistic or plausible.
- It cannot be used for continuous variables.
- It does not take into account the correlation between variables.

## Comparison and Contrast

The choice of imputation strategy depends on the nature of the missing data and the goal of the analysis. Here are some factors to consider when comparing and contrasting imputation strategies:

- Mean imputation is best suited for continuous data that is approximately normally distributed, while median imputation is best suited for continuous data that is skewed.
- Mode imputation is best suited for categorical data or non-parametric data that does not have a clear mean or median.
- Mean imputation assumes that the missing values are MAR and does not work well when the missing values are not MAR. Median and mode imputation do not assume that the missing values are MAR and can work well when the missing values are not MAR.
- Mean imputation can be affected by outliers, while median imputation is robust to outliers.
- Mode imputation may lead to bias if the most frequent value is not representative of the true underlying distribution.

In general, it is important to choose an imputation strategy that is appropriate for the type of data and the nature of the missing values. It is also important to evaluate the imputation results and assess the impact of imputation on the analysis.
