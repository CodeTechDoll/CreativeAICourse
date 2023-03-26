# Downloading and Installing a Dataset

Here are the steps to download and install a dataset for use in your project:

## Step 1: Find a Dataset

Find a suitable dataset for your project. Some popular sources for datasets include:

- [Kaggle](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

Choose a dataset that aligns with your project goals and requirements.

## Step 2: Download the Dataset

Once you've selected a dataset, download it to your local machine. Datasets typically come in various file formats, such as CSV, JSON, or Excel. For this example, we'll assume you're working with a CSV file.

## Step 3: Place the Dataset in Your Project Directory

Create a new directory in your project called data and move the downloaded dataset into it. This keeps your project organized and makes it easier to access the dataset in your Python scripts.

```python
ai_course/
│
├── .gitignore
├── README.md
├── data/
│   └── your_dataset.csv
│
├── unit_1/
│   ├── README.md
│   ├── python_files/
│   │   ├── example_1.py
│   │   └── example_2.py
│   └── images/
│       ├── image_1.png
│       └── image_2.png
...
```

Replace your_dataset.csv with the name of your downloaded dataset.

### If you want to replicate my results, I am using the "Gender Pay Gap - Europe (2010 - 2021)" dataset from Kraggle

## Step 4: Load the Dataset in Your Python Script

In your Python script, use the pandas library to load the dataset. Make sure to adjust the file path and name to match your dataset:

```python

import pandas as pd

# Load the dataset
data = pd.read_csv('data/your_dataset.csv')

# Display the first few rows of the dataset
print(data.head())
```

Replace 'data/your_dataset.csv' with the correct path and name of your dataset.

Now you have successfully downloaded and installed a dataset for use in your project. You can proceed with data preprocessing, analysis, and model training.
