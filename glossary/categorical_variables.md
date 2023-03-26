# Categorical Variables

Categorical variables are a type of variable that can take on one of a limited, and usually fixed, number of possible categories or levels. They represent qualitative properties or characteristics of an object or an observation. Categorical variables can be further divided into two types:

- Ordinal categorical variables: These have a natural order or ranking to the categories, such as 'low', 'medium', and 'high'. While there is an order to the levels, the difference between the levels is not necessarily equal or meaningful.
- Nominal categorical variables: These do not have any natural order or ranking to the categories, such as colors ('red', 'blue', 'green') or ice-cream flavor ('chocolate', 'vanilla', 'moose-tracks', 'rocky road').

## Encoding

Encoding categorical variables means converting the categories into a numerical form that can be used by machine learning algorithms. Most machine learning algorithms work with numerical values and cannot process categorical data directly. Encoding categorical variables allows these algorithms to use the information contained in the categorical variables effectively.

There are several common techniques for encoding categorical variables, including:

- **Label encoding**: Assigning a unique integer value to each category. This method is more suitable for ordinal categorical variables since it introduces an order to the categories.
- **One-hot encoding**: Creating binary columns for each category, where each binary column indicates the presence (1) or absence (0) of a category for an observation. This method is suitable for nominal categorical variables as it does not impose any order on the categories.
- **Target encoding**: In target encoding, you replace each category with the mean of the target variable for that category. This method is especially useful when dealing with high-cardinality categorical variables, as it reduces the dimensionality of the dataset while preserving the relationship between the categorical variable and the target variable. However, it may introduce leakage if not implemented carefully, so it's crucial to calculate the mean values separately for the training and validation/test sets.
- **Binary encoding**: Binary encoding converts the integer codes assigned to the categories into binary numbers and then creates new binary columns for each bit in the binary representation. This method is a compromise between one-hot encoding and label encoding, as it reduces the dimensionality of the dataset while not introducing an artificial order among the categories.
- **Embeddings:** Embeddings are another approach to handling high-cardinality categorical variables, particularly in deep learning models. An embedding is a dense vector representation of the categorical variable that captures the relationships between the categories in a lower-dimensional space. You can either pre-train an embedding using unsupervised techniques or learn the embedding as part of your model training process.
-

If you have a nominal categorical variable that is not binary, like "country," one-hot encoding is still an appropriate choice for encoding. However, keep in mind that one-hot encoding can lead to a high-dimensional dataset if there are many unique categories (e.g., many countries), which might result in increased memory usage and longer training times for machine learning models.

When using one-hot encoding for a variable like "country," you would create a new binary column for each unique country in the dataset. For example, if you have data on three countries—USA, UK, and Canada—your one-hot encoded dataset would have three new binary columns, one for each country. Each row in the dataset would have a 1 in the column corresponding to the country of the observation and 0s in the other columns.

It is essential to choose the appropriate encoding technique based on the nature of the categorical variable and the requirements of the specific machine learning algorithm being used.

## Ordinal vs. Nominal Categorical Variables

**Ordinal categorical variables** have an inherent order or ranking among the categories. The order is meaningful and can provide valuable information when analyzing the data or making predictions. Examples of ordinal categorical variables include:

- Education levels (elementary, high school, undergraduate, graduate)
- Customer satisfaction ratings (very dissatisfied, dissatisfied, neutral, satisfied, very satisfied)
- T-shirt sizes (small, medium, large, extra-large)

It's important to note that while there is a natural order to the categories in ordinal variables, the difference between the categories might not be consistent or quantifiable.

**Nominal categorical variables**, on the other hand, do not have any natural order among their categories. They represent distinct classes or groups with no inherent ranking. Examples of nominal categorical variables include:

- Colors (red, blue, green, yellow)
- Animal species (dog, cat, bird, fish)
- Marital status (single, married, divorced, widowed)

### Categorical Variables in Machine Learning

Categorical variables play a crucial role in machine learning as they often represent essential features or characteristics of the data. However, most machine learning algorithms work with numerical values and cannot process raw categorical data directly. As a result, it's necessary to encode categorical variables into a numerical form.

Encoding techniques should be chosen based on the nature of the categorical variable (ordinal or nominal) and the requirements of the specific machine learning algorithm being used.

### Choosing the Right Encoding Technique

#### Label Encoding

Label encoding assigns an integer to each unique category in a variable. It's suitable when:

- The categorical variable is ordinal, meaning it has a natural order (e.g., "low", "medium", "high").
- The number of categories is relatively small and their cardinality is low.
- The algorithm you're using can handle ordinal data properly (e.g., decision trees).

#### One-Hot Encoding

One-hot encoding creates a binary feature for each unique category in a variable. It's suitable when:

- The categorical variable is nominal, meaning it doesn't have a natural order (e.g., "red", "blue", "green").
- The number of categories is relatively small, and their cardinality is low.
- The algorithm you're using requires numerical input data (e.g., logistic regression, neural networks).

#### Target Encoding

Target encoding replaces a category with the mean of the target variable for that category. It's suitable when:

- The categorical variable is nominal, and there's a relationship between the category and the target variable.
- The number of categories is large, or their cardinality is high.
- The algorithm you're using can handle ordinal data (e.g., decision trees).

> Note: Target encoding can introduce leakage if not done properly. Always compute the encoding separately for your training and validation/test sets to avoid leakage.
>
#### Binary Encoding

Binary encoding converts each category's integer code (from label encoding) to its binary representation and creates a separate feature for each bit. It's suitable when:

- The categorical variable is nominal.
- The number of categories is relatively large, or their cardinality is high.
- The algorithm you're using requires numerical input data.

#### Embeddings

Embeddings represent categories as continuous vectors in lower-dimensional spaces. They are suitable when:

- The categorical variable is nominal.
- The number of categories is very large, or their cardinality is high (e.g., user IDs, product IDs).
- You're working with deep learning models, such as neural networks, which can learn embeddings as part of the model training process.

To choose the right encoding method, consider the nature of the categorical variable (ordinal or nominal), the number of unique categories, their cardinality, and the requirements of the machine learning algorithm you're using. Be aware of the potential for leakage when using methods like target encoding and always validate your model's performance using a separate validation or test set.
