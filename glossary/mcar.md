# Missing Completely At Random (MCAR)

## Summary

In statistics, missing completely at random (MCAR) is a term used to describe a missing data mechanism that occurs when the missingness of the data is unrelated to both the observed and unobserved data. This means that the probability of a data point being missing is unrelated to the value of the missing variable itself or any other variable(s) in the dataset.

For example, consider a survey that asks participants to report their age, gender, and height. Suppose that some participants refuse to answer some questions, and the probability of a question being left unanswered is the same for all participants, regardless of their age, gender, or height. In this case, the missingness is considered to be MCAR because the probability of a value being missing is unrelated to any other variable in the dataset.

## Example

Suppose you have a dataset containing the following variables:

- Age: continuous variable representing the age of individuals.
- Income: continuous variable representing the income of individuals.

In this dataset, there are some missing values for income, but the missingness is completely random. Specifically, the probability of a value being missing is unrelated to both the actual income value and the age of the individual.

Here's some Python code that generates a random dataset with missing values that are MCAR:

```python
import pandas as pd
import numpy as np

# generate random data
data = pd.DataFrame({
    'age': np.random.randint(20, 60, size=100),
    'income': np.random.choice([np.nan, 50000, 60000, 70000], size=100, p=[0.2, 0.2, 0.2, 0.2])
})

# check correlation between missingness and variables
print(data.corr())
```

In this example, we generate a dataset of 100 individuals with ages ranging from 20 to 60 years old. We randomly assign missing values for income, where 20% of the values are missing. We also set the missingness probabilities to be independent of any other variables, such as age, and instead use a probability distribution with equal probabilities for each value.

We can then check the correlation between the missingness and variables using the corr() function. If the correlation is close to zero, it indicates that the missingness is unrelated to any other variables in the dataset, and therefore, the missingness is MCAR.
