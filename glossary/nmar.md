# Not Missing At Random (NMAR)

In statistics, not missing at random (NMAR) is a term used to describe a missing data mechanism that occurs when the missingness of the data is related to the value of the missing variable itself or some other variable(s) in the dataset. This means that the probability of a data point being missing depends on the value of the missing variable or some other variable(s) in the dataset.

For example, consider a survey that asks participants to report their income and their satisfaction with their job. Suppose that some participants refuse to answer the question about their income, and the probability of a value being missing depends on their income level. Specifically, individuals with low incomes are more likely to refuse to answer the question about their income. In this case, the missingness is considered to be NMAR because the probability of a value being missing is related to the value of the missing variable itself.

## Example

Suppose you have a dataset containing the following variables:

- Age: continuous variable representing the age of individuals.
- Income: continuous variable representing the income of individuals.
- Satisfaction: categorical variable representing the satisfaction level of individuals with their job.

In this dataset, there are some missing values for income, and the missingness is not random. Specifically, individuals who are less satisfied with their job are more likely to have missing income data than individuals who are more satisfied. In this case, the missingness is considered to be NMAR because the probability of a value being missing depends on the value of the satisfaction variable, which is not observed.

Here's some Python code that generates a random dataset with missing values that are NMAR:

```python
import pandas as pd
import numpy as np

# generate random data
data = pd.DataFrame({
    'age': np.random.randint(20, 60, size=100),
    'income': np.random.choice([np.nan, 50000, 60000, 70000], size=100, p=[0.2, 0.2, 0.2, 0.2]),
    'satisfaction': np.random.choice(['low', 'medium', 'high'], size=100, p=[0.3, 0.4, 0.3])
})

# set missingness probabilities based on satisfaction level
data.loc[data['satisfaction'] == 'low', 'income'] = np.nan
data.loc[data['satisfaction'] == 'medium', 'income'] = np.random.choice([50000, 60000], size=sum(data['satisfaction'] == 'medium'))
data.loc[data['satisfaction'] == 'high', 'income'] = np.random.choice([50000, 60000, 70000], size=sum(data['satisfaction'] == 'high'))

# check correlation between missingness and variables
print(data.corr())
```

In this example, we generate a dataset of 100 individuals with ages ranging from 20 to 60 years old. We randomly assign missing values for income, but the probability of a value being missing depends on the value of the satisfaction variable. Specifically, individuals who report a low level of job satisfaction are more likely to have missing income data, while individuals who report a high level of job satisfaction are more likely to have income data. We set missingness probabilities based on the satisfaction level, such that individuals with low satisfaction have missing income data, and individuals with high satisfaction are more likely to have income data.

We can then check the correlation between the missingness and variables using the corr() function. If the correlation is non-zero, it indicates that the missingness is related to some other
