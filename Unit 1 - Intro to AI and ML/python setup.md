# Setting Up Python and Required Libraries

Follow the steps below to set up your Python environment and install the necessary libraries for the project.

## Step 1: Install Python and VSCode

If you haven't already, download and install the latest version of Python from the official website. Choose the version appropriate for your operating system (Windows, macOS, or Linux).

Additionally, install VSCode into a directory of your choosing. This will be our IDE of choice for most of this course.

## Step 2: Set Up a Virtual Environment (Optional)

It's a good practice to create a virtual environment for each project to keep the dependencies organized and avoid conflicts with system-wide packages. To create a virtual environment, follow these steps:

- Open a terminal or command prompt.

- Navigate to the directory where you want to create your project.

- Run the following command to create a virtual environment:

```bash
python -m venv your_venv_name
```

Replace your_venv_name with a name of your choice.

Activate the virtual environment:

### On Windows

```bash
your_venv_name\Scripts\activate
```

### On macOS and Linux

```bash
source your_venv_name/bin/activate
```

## Step 3: Install Required Libraries

To install the required libraries, run the following command in your terminal or command prompt:

```bash
pip install pandas scikit-learn matplotlib
```

This will install the following libraries:

- pandas: A library for data manipulation and analysis.
- scikit-learn: A library for machine learning and data mining.
- matplotlib: A library for creating static, animated, and interactive visualizations in Python.

## Step 3.5: Use Requirements

Python allows you to "freeze" the requirements/modules needed to run your project
You can export the current requirements with the command

```bash
pip freeze > requirements.txt
```

And then install the associated requirements with

```bash
pip install -r requirements.txt
```

## Step 4: Import Libraries in Your Python Script

At the beginning of your Python script, add the following lines to import the necessary libraries:

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
```

Now your Python environment is set up, and you can proceed with implementing your project.
