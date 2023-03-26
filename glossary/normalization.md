# Normalization

Normalization is a data pre-processing technique used in machine learning to scale numerical features to a common range. The purpose of normalization is to bring all features to a similar scale so that no single feature dominates the others. This can be important because many machine learning algorithms use distance-based metrics to measure similarity between data points, and features with larger scales can dominate the distance calculations.

Normalization involves transforming the values of a feature to a new range, typically between 0 and 1. There are several methods for normalization, including:

## Min-max normalization

Min-max normalization scales the values of a feature to a range between 0 and 1, using the minimum and maximum values of the feature. The formula for min-max normalization is:

```x_normalized = (x - x_min) / (x_max - x_min)```
where x is the original value of the feature, x_min is the minimum value of the feature, and x_max is the maximum value of the feature. This formula scales the value of x to a range between 0 and 1.

Min-max normalization is a common method for normalizing features, and is appropriate when the minimum and maximum values of the feature are known.

When we have numerical features with different scales or ranges, some features may dominate others in machine learning models that use distance-based metrics or regularization. Min-max normalization can help to prevent this by scaling the features to a common range.

### Example

Here's an example in Python using scikit-learn to perform min-max normalization:

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# create a sample feature matrix
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# normalize the feature matrix using min-max normalization
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

print(X_normalized)
```

In this example, we create a sample feature matrix X. We then normalize the feature matrix using scikit-learn's MinMaxScaler() class. The resulting X_normalized matrix has each feature scaled to a range between 0 and 1.

### When to Use

Min-max normalization is commonly used in image processing tasks, where the pixel values of an image are scaled to a fixed range between 0 and 1. It is also used in recommender systems, where we want to scale the ratings of items to a common range.

However, min-max normalization may not be appropriate for all situations. In particular, it can be sensitive to outliers in the data, as the scaling is determined by the minimum and maximum values of the feature. It may also not be appropriate if the distribution of the data is highly skewed or if there are extreme values in the data. In such cases, other normalization techniques, such as z-score normalization or L2 normalization, may be more appropriate.

### Example

Let's say you are building a machine learning model to predict the price of a house based on its features, such as the number of bedrooms, the size of the lot, and the location of the house.

The features of the dataset are on different scales, where the number of bedrooms is between 1 and 6, the size of the lot is between 500 and 10,000 square feet, and the location is represented as a categorical variable with 5 possible values.

Before building the machine learning model, you decide to normalize the features using min-max normalization.

By normalizing the features, you can ensure that each feature is on the same scale and that the differences between the values of the features are preserved. This can help to improve the performance of the machine learning model, as it can reduce the impact of features with large ranges on the final predictions.

Here's an example in Python using scikit-learn to perform min-max normalization on a dataset of house features:

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# create a sample dataset of house features
X = np.array([
    [3, 2000, 0],
    [4, 1500, 2],
    [2, 10000, 1],
    [5, 5000, 4],
    [1, 3000, 3]
])

# normalize the dataset using min-max normalization
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

print(X_normalized)
```

In this example, we create a sample dataset X of house features. We then normalize the dataset using scikit-learn's MinMaxScaler() class. The resulting X_normalized matrix has each feature scaled to a range between 0 and 1.

### Summary

It is useful in situations where we want to ensure that all numerical features are on the same scale, and where we want to preserve the relative differences between the values of the features.

One advantage of min-max normalization is that it is a simple and intuitive method that is easy to implement. It can be particularly useful when the distribution of the features is unknown or when there are outliers in the data, as it does not make any assumptions about the distribution of the data.

Another advantage of min-max normalization is that it preserves the relative relationships between the values of the features. For example, if one feature has values that are twice as large as another feature, the normalized values will also reflect this relationship.

## Z-score normalization

Z-score normalization scales the values of a feature to have a mean of 0 and a standard deviation of 1. The formula for z-score normalization is:

```x_normalized = (x - mu) / sigma``
where x is the original value of the feature, mu is the mean of the feature, and sigma is the standard deviation of the feature. This formula scales the value of x to have a mean of 0 and a standard deviation of 1.

Z-score normalization is useful when the distribution of the feature is not known, or when there are outliers in the data.

## Decimal scaling normalization

Decimal scaling normalization scales the values of a feature by dividing them by a power of 10, such that the absolute value of the largest value of the feature is less than 1. The formula for decimal scaling normalization is:

```x_normalized = x / (10^j)```
where x is the original value of the feature, and j is the smallest integer such that max(|x_normalized|) < 1.

Decimal scaling normalization is useful when the absolute values of the features are very large or very small, and can help improve the convergence of some machine learning algorithms.

## L2 Normalization

L-2 normalization, also known as Euclidean normalization or vector normalization, is another method of normalization that scales the values of a feature vector to a unit vector of length 1.

L-2 normalization is used when the direction of the feature vector is more important than the magnitude of the values. It is commonly used in text classification, image recognition, and recommender systems.

The formula for L-2 normalization is:

`x_normalized = x / ||x||`
where x is the original feature vector, and ||x|| is the L-2 norm of the vector, defined as:

`||x|| = sqrt(sum(xi^2))`
where xi is the i-th element of the feature vector.

L-2 normalization scales each feature value by the magnitude of the feature vector, such that the resulting vector has a length of 1. This has the effect of emphasizing the direction of the vector and de-emphasizing the magnitude of the values.

Here's an example in Python using scikit-learn to perform L-2 normalization:

```python
from sklearn.preprocessing import normalize
import numpy as np

# create a sample feature matrix
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# normalize the feature matrix using L-2 normalization
X_normalized = normalize(X, norm='l2')

print(X_normalized)
```

In this example, we create a sample feature matrix X. We then normalize the feature matrix using scikit-learn's normalize() function with the norm='l2' parameter, which specifies L-2 normalization. The resulting X_normalized matrix has each feature vector scaled to a unit vector of length 1.

### Further Explanation

L-2 normalization is a way of scaling a set of numbers so that they form a unit vector. A unit vector is a vector with a length of 1.

For example, suppose we have a set of three numbers: 3, 4, and 5. We can represent this set of numbers as a vector in three-dimensional space, where the first number corresponds to the x-coordinate, the second number corresponds to the y-coordinate, and the third number corresponds to the z-coordinate. So, our vector would be (3, 4, 5).

To normalize this vector using L-2 normalization, we first calculate the length of the vector using the formula:
`length = sqrt(3^2 + 4^2 + 5^2) = sqrt(50) ≈ 7.07`

This length is also called the L-2 norm of the vector.

We then divide each number in the vector by the length, like this:

`(3 / 7.07, 4 / 7.07, 5 / 7.07) ≈ (0.424, 0.566, 0.707)`

This resulting vector is a unit vector, because its length is 1:

`sqrt(0.424^2 + 0.566^2 + 0.707^2) ≈ 1`

So, L-2 normalization scales the original set of numbers so that they form a unit vector with the same direction as the original vector, but with a length of 1.

### When Would I Use This?

L-2 normalization is commonly used in machine learning tasks that involve measuring the similarity between vectors, such as document classification, image recognition, and recommendation systems.

#### Example

For example, suppose we have a set of documents that we want to classify based on their content. We can represent each document as a vector, where each element of the vector corresponds to a term in the document, and the value of the element indicates the frequency of the term in the document.

To measure the similarity between two documents, we can compute the cosine similarity between their corresponding vectors. The cosine similarity is a measure of the angle between two vectors, and it ranges from -1 (opposite directions) to 1 (same direction).

However, before computing the cosine similarity, we need to normalize the vectors using L-2 normalization to ensure that the length of the vector does not affect the similarity measure.

Without L-2 normalization, documents with more terms or longer text could have higher vector lengths and appear more similar, even if their content is not actually similar. By using L-2 normalization, we can ensure that the similarity measure is based on the direction of the vectors, rather than their length.

Here's an example in Python using scikit-learn to compute cosine similarity between two L-2 normalized vectors:

```python
from sklearn.preprocessing import normalize
import numpy as np

# create two sample vectors
x = np.array([3, 4, 5])
y = np.array([1, 2, 3])

# normalize the vectors using L-2 normalization
x_normalized = normalize(x.reshape(1, -1), norm='l2')
y_normalized = normalize(y.reshape(1, -1), norm='l2')

# compute the cosine similarity between the normalized vectors
cosine_sim = np.dot(x_normalized, y_normalized.T)

print(cosine_sim)
```

In this example, we create two sample vectors x and y. We then normalize the vectors using scikit-learn's normalize() function with the norm='l2' parameter, which specifies L-2 normalization. We then compute the cosine similarity between the normalized vectors using numpy's dot() function. The resulting cosine_sim is a measure of the similarity between the two vectors, based on their direction rather than their length.

#### TODO: Improve this example

#### Another Example

 Let's say you are working on a natural language processing project where you want to compare the similarity between two documents. You decide to use L-2 normalization to normalize the vectors representing the documents, so that their lengths do not affect the similarity measure.

For example, suppose you have two documents:

```
Document 1: "The quick brown fox jumped over the lazy dog."
Document 2: "The lazy dog was jumped over by the quick brown fox."
```

You first represent each document as a vector, where each element corresponds to a term in the document and the value of the element indicates the frequency of the term in the document. For simplicity, let's assume that we only consider 6 terms in the documents: "quick", "brown", "fox", "jumped", "lazy", and "dog". The vectors for the two documents are:

```
Document 1 vector: (1, 1, 1, 1, 1, 1)
Document 2 vector: (1, 1, 1, 1, 1, 1)
```

These vectors have the same length, because both documents contain the same number of terms. However, we want to measure the similarity between the documents based on the direction of the vectors, not their length.

To do this, we can use L-2 normalization to normalize the vectors to a unit vector with a length of 1. The normalized vectors are:

```
Document 1 normalized vector: (0.408, 0.408, 0.408, 0.408, 0.408, 0.408)
Document 2 normalized vector: (0.408, 0.408, 0.408, 0.408, 0.408, 0.408)
```

Now we can measure the cosine similarity between the two normalized vectors to determine their similarity. The cosine similarity is a measure of the angle between the two vectors, and it ranges from -1 (opposite directions) to 1 (same direction). In this case, the cosine similarity between the two vectors is 1, indicating that they are identical.

L-2 normalization can help to ensure that our document similarity measure is based on the content of the documents, rather than their length or size.

#### Conclusion

L-2 normalization is a technique for removing the influence of the length or magnitude of a vector when comparing it to other vectors.

When we use machine learning to compare or measure the similarity between vectors, the length of a vector can sometimes overshadow its other important characteristics. By normalizing the vector using L-2 normalization, we can ensure that the direction of the vector is what is being compared or measured, rather than its length.

This technique is often used in natural language processing tasks, such as document classification or recommendation systems, where the length of a document can influence its similarity to other documents, even if their content is dissimilar. L-2 normalization allows us to compare documents based on their content, rather than their length or size.
