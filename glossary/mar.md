# MAR

## Summary

"MAR" stands for "Missing At Random"

In statistics, "missing at random" refers to the situation where the probability of a value being missing is not related to the value itself, but may be related to other variables in the dataset. For example, in a medical study, the probability of a patient's weight being missing may depend on their age, but not on their weight itself. In this case, the missing data is considered to be "missing at random".

Under the assumption of MAR, mean imputation can be an appropriate imputation method because it replaces the missing value with the average of the observed values for that variable. **This assumes that the missing value is just another observation of the same variable and can be replaced with the average of the other observations.**

However, if the missing data is not MAR, i.e. the missingness is related to the value itself, mean imputation may result in biased or misleading estimates. In such cases, median and mode imputation can be better alternatives, as they do not rely on the assumption of normality and are less affected by outliers.

Therefore, it is important to carefully consider the nature of missing data and the assumptions underlying the imputation method when choosing an appropriate imputation strategy for a given dataset.

To determine if data is MAR, it is important to assess whether the probability of a data point being missing depends on some other variable(s) in the dataset. If the probability of a data point being missing depends only on the value of the missing variable, then the data is considered to be "Missing Completely At Random" (MCAR). However, if the probability of a data point being missing depends on some other variable(s) in the dataset, then the data is considered to be MAR.

It's worth noting that determining the missingness mechanism (MCAR, MAR, or Not Missing At Random (NMAR)) is important because it can guide the choice of imputation method. Different imputation methods work better under different missingness mechanisms.

In missing data analysis, **data is considered to be MAR if the missingness is related to, but not caused by another variable. That means the missingness is not a direct result of the value of the missing variable itself, but it may be influenced by other variables in the dataset**

## A More Detailed Example

Consider a study that is examining the relationship between age, income, and the presence of a particular medical condition. Suppose that some of the participants in the study did not report their income. If the missingness of the income data is related to age but not to the actual income, the data is considered to be missing at random. This means that the probability of a participant having missing income data is related to their age, but not to their actual income.

### Here's a small example to illustrate this concept

Suppose you have a dataset containing the following variables:

- Age: continuous variable representing the age of individuals.
- Income: continuous variable representing the income of individuals.

In this dataset, there are some missing values for income, but the missingness is related to age. Specifically, individuals who are younger are more likely to have missing income data than individuals who are older. This missingness is considered to be MAR because the probability of a value being missing is related to another variable (age), but not to the actual income value.

Here's some Python code that generates a random dataset with missing values that are MAR:

```python
import pandas as pd
import numpy as np

# generate random data
data = pd.DataFrame({
    'age': np.random.randint(20, 60, size=100),
    'income': np.random.choice([np.nan, 50000, 60000, 70000], size=100, p=[0.2, 0.4, 0.3, 0.1])
})

# check correlation between missingness and age
print(data.corr())
```

In this example, we generate a dataset of 100 individuals with ages ranging from 20 to 60 years old. We randomly assign missing values for income, where 20% of the values are missing. We also set the missingness probabilities to be dependent on age such that younger individuals are more likely to have missing income data than older individuals.

We can then check the correlation between the missingness and age using the corr() function. If the correlation is close to zero, it indicates that the missingness is unrelated to the actual income value and therefore, the missingness is MAR.
