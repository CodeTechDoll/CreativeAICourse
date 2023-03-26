# DataFrame

A DataFrame is a two-dimensional, size-mutable, and potentially heterogeneous tabular data structure with labeled axes (rows and columns) in the pandas library, a popular data manipulation and analysis library in Python. A DataFrame can be thought of as a table of data, similar to a spreadsheet or a SQL table. It allows you to easily manipulate and analyze data by providing various built-in functions for cleaning, transforming, aggregating, and visualizing data.

A DataFrame can be created from various data sources, such as:

    Lists
    Dictionaries
    CSV files
    Excel files
    SQL queries

Here's an example of creating a DataFrame from a dictionary:

```python
import pandas as pd

# Create a dictionary of data
data = {
    'Country': ['USA', 'UK', 'Canada', 'USA', 'Canada'],
    'Population': [331, 68, 38, 331, 38],
    'Area': [9834, 243, 9985, 9834, 9985]
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Display the DataFrame
print(df)
```
