# SKLearn Models

Scikit-learn is a popular machine learning library in Python, and it includes various machine learning models for different types of problems. Here are some of the main models from scikit-learn:

## Linear Regression

Linear Regression is used for regression problems, where the target variable is continuous. It finds the best fit line that relates the input variables to the target variable.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions using the model
y_pred = model.predict(X_test)

# Calculate the metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)
```

## Logistic Regression

Logistic Regression is used for binary classification problems, where the target variable has only two classes. It models the probability of a binary outcome.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Initialize the model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions using the model
y_pred = model.predict(X_test)

# Calculate the metrics
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
```

## K-Nearest Neighbors

K-Nearest Neighbors is used for both regression and classification problems. It works by finding the K-nearest data points to a new data point and using their values to make a prediction.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Initialize the KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions using the model
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```

## Decision Trees

Decision Trees is used for both regression and classification problems. It creates a tree-like model of decisions and their possible consequences, and it can be used to make predictions on new data.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Initialize the model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions using the model
y_pred = model.predict(X_test)

# Calculate the metrics
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
```

## Random Forest

Random Forest is an ensemble method that combines multiple decision trees to create a more accurate model. It is used for both regression and classification problems.

```py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Initialize the model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions using the model
y_pred = model.predict(X_test)

# Calculate the metrics
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
```

## Support Vector Machines (SVM)

SVM is used for both regression and classification problems. It finds the best separating hyperplane between the data points in a high-dimensional space.

```py
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Initialize the model
model = SVC()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions using the model
y_pred = model.predict(X_test)

# Calculate the metrics
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
```

## Naive Bayes

Naive Bayes is used for classification problems. It is based on Bayes' theorem and assumes that the input variables are independent of each other.

```py
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Initialize the Naive Bayes model
model = GaussianNB()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions using the model
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Neural Networks

Neural Networks are used for both regression and classification problems. They are based on the structure of the human brain and can learn complex relationships between input variables and the target variable.

These are just some of the main models from scikit-learn, and there are many more to explore depending on the type of problem you are trying to solve.

```py
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Initialize the Neural Network model
model = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions using the model
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
