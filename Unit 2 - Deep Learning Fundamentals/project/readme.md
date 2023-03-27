# Project

## Build a simple neural network to classify handwritten digits using the MNIST dataset

1. **Train the neural network using backpropagation and a suitable optimizer**

2. **Evaluate the performance of the neural network on the test set**

3. **Experiment with different network architectures and compare their performance**

4. **Document your findings**

After completing this homework, you should have a better understanding of how neural networks work, how to design and train them, and how to evaluate their performance on a classification task.

### 1. Load the MNIST dataset and preprocess the data

- [Loading the MNIST dataset in Keras](https://keras.io/api/datasets/mnist/)
- [Normalize the data](https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/)
- [Split the data into train and test sets](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

#### Goals

- Understand the MNIST dataset and its structure
- Learn how to load and preprocess data for deep learning

#### Key Concepts

- MNIST dataset
- Data normalization
- Train-test split

#### MNIST Dataset

The MNIST (Modified National Institute of Standards and Technology) dataset is a collection of 70,000 handwritten digits from 0 to 9. It has become a standard benchmark for evaluating the performance of machine learning algorithms, particularly in the domain of image recognition. The dataset is split into a training set of 60,000 images and a test set of 10,000 images. Each image is grayscale and 28x28 pixels in size.

#### Data Normalization

Data normalization is a preprocessing step that scales the input features to be within a similar range of values. This is important because having features with different scales can cause the optimization algorithm to converge slowly or become stuck in local minima. In the context of the MNIST dataset, pixel values range from 0 to 255. We can normalize these values by dividing each pixel value by 255, resulting in a range of 0 to 1 for all pixel values.

```python
train_images, test_images = train_images / 255.0, test_images / 255.0
```

#### Train-test Split

In machine learning, it is common practice to split the dataset into a training set and a test set. The training set is used to train the model, while the test set is used to evaluate its performance. This separation allows us to assess how well our model generalizes to unseen data, which is a critical aspect of evaluating its effectiveness.

The MNIST dataset comes pre-split into a training set of 60,000 images and a test set of 10,000 images. When loading the dataset using TensorFlow or Keras, these sets can be easily accessed:

```python
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

Now that you have an understanding of the MNIST dataset, data normalization, and train-test split, you are ready to move on to the next steps of designing and training a neural network to classify the handwritten digits.

```python
# Load the MNIST dataset
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# No need to split the data into train and test sets, as it's already provided by Keras
```

### 2. Design a Neural Network Architecture

#### Goals

- Understand the components of a neural network architecture
- Learn how to design a simple neural network using Keras

#### Key Concepts

- Neural network layers (Dense, Convolutional, etc.)
- Activation functions

#### Keras Sequential Model

The `Sequential` model is a linear stack of layers, where you can simply add one layer at a time. It is the most straightforward and common way to create neural networks in Keras.

#### Neural Network Layers

Keras provides various types of layers to build a neural network architecture. Some of the most common layer types are:

- `Dense`: A fully connected layer where each neuron is connected to every neuron in the previous layer. This layer is commonly used in feedforward neural networks.
- `Conv2D`: A convolutional layer that applies filters to the input, useful for image processing and feature extraction in convolutional neural networks (CNNs).
- `MaxPooling2D`: A pooling layer that reduces the spatial dimensions of the input, often used in CNNs to reduce the number of parameters and control overfitting.
- `LSTM`: Long Short-Term Memory layer, a type of recurrent layer used in recurrent neural networks (RNNs) to model sequential data and learn long-term dependencies.
- `GRU`: Gated Recurrent Unit layer, another type of recurrent layer with a simplified architecture compared to LSTM, often used in RNNs for similar purposes.

#### Activation Functions

Activation functions are used to introduce non-linearity in the neural network. Some common activation functions are:

- `ReLU`: Rectified Linear Unit, a widely used activation function defined as `f(x) = max(0, x)`. It's computationally efficient and helps mitigate the vanishing gradient problem.
- `Sigmoid`: A function that maps the input to a value between 0 and 1, often used for binary classification tasks in the output layer.
- `Softmax`: A function that converts a vector of logits into probabilities, often used in the output layer for multi-class classification tasks.
- `Tanh`: Hyperbolic tangent function, similar to the sigmoid function but maps input to a range between -1 and 1. It's sometimes used in hidden layers of RNNs.

#### Example

In the following example, we'll design a simple neural network for the MNIST dataset using the Keras Sequential model.

```python
from keras.models import Sequential
from keras.layers import Dense, Flatten

# -- rest of code --

# Design a simple neural network architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the input images
model.add(Dense(128, activation='relu'))  # Add a Dense layer with 128 neurons and ReLU activation
model.add(Dense(10, activation='softmax'))  # Add an output layer with 10 neurons and softmax activation
```

In this example, we first create a `Sequential` model, and then add a `Flatten` layer to convert the input images into a flat vector. Next, we add a `Dense` layer with 128 neurons and `ReLU` activation, followed by another `Dense` layer with 10 neurons and softmax activation for the output. This simple architecture is suitable for classifying the MNIST dataset of handwritten digits.

### 3. Train the neural network using backpropagation and a suitable optimizer

- [Keras Model Training](https://keras.io/guides/training_with_built_in_methods/)
- [Optimizers (Stochastic Gradient Descent, Adam, etc.)](https://keras.io/api/optimizers/)

#### Goals

- Learn how to train a neural network using Keras
- Understand the role of backpropagation and optimizers

#### Key Concepts

- Backpropagation
- Optimizers (Stochastic Gradient Descent, Adam, etc.)
Keras makes it easy to train a neural network with a simple, high-level API. The training process involves two main steps:
- Compiling the model, which configures the model's optimizer, loss function, and evaluation metrics.
- Fitting the model to the training data using the `fit()` method.

#### Backpropagation

#### Optimizers (Stochastic Gradient Descent, Adam, etc.)

Optimizers are algorithms that adjust the weights of a neural network during training to minimize the loss function. There are various optimization algorithms available, and each has its own strengths and weaknesses. Some popular optimizers include:

- Stochastic Gradient Descent (SGD)
- Adam (Adaptive Moment Estimation)
- RMSprop (Root Mean Square Propagation)
- Adagrad (Adaptive Gradient Algorithm)

When compiling a model in Keras, you can choose an optimizer by specifying its name as the `optimizer` argument. For example, you can use the Adam optimizer by setting `optimizer='adam'`.

```python
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the neural network
model.fit(train_images, train_labels, epochs=10)
```

In this code snippet, the model is compiled with the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric. Sparse categorical crossentropy is a variant of categorical crossentropy that doesn't require one-hot encoding for the labels.

In this example, the model.fit function trains the neural network using backpropagation. The optimizer (specified as 'adam' in the model.compile function) is responsible for updating the weights during training, while the backpropagation algorithm computes the gradients of the loss function with respect to each weight. The `epochs` parameter in the `model.fit` function determines the number of times the entire training dataset will be used for updating the weights.

The model is then trained using the `fit()` method, which takes the training images and labels as input, along with the number of training epochs. The `fit()` method uses backpropagation to adjust the weights of the neural network during training, minimizing the loss function over time.

During training, the optimizer updates the weights of the neural network based on the gradients of the loss function with respect to the weights. This process is known as **backpropagation**, and it's the core of training deep neural networks.

Different optimizers have different strategies for updating the weights, but they all aim to minimize the loss function efficiently. The choice of optimizer can significantly affect the training process and the final performance of the neural network, so it's important to understand their properties and select a suitable optimizer for the task at han

### 4. Evaluate the performance of the neural network on the test set

- [Keras Model Evaluation](https://keras.io/guides/training_with_built_in_methods/#evaluating-a-model)
- [Accuracy](https://keras.io/api/metrics/accuracy_metrics/)
- [Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

#### Goals

- Understand how to evaluate a neural network's performance
- Learn how to use Keras and scikit-learn to compute evaluation metrics

#### Key Concepts

- Model evaluation
- Accuracy
- Confusion matrix

Evaluating a trained neural network is a critical step to ensure its performance and generalization capabilities. It helps in understanding whether the model is overfitting or underfitting and, if necessary, fine-tuning the model's hyperparameters.

#### Model Evaluation

Keras provides a simple method, `evaluate`, for computing the performance of a trained model on a test dataset. The `evaluate` function returns the loss value and any other metrics specified during the model's compilation.

```py
test_loss, test_acc = model.evaluate(test_images, test_labels)
```

##### Accuracy

Accuracy is a common metric for classification tasks. It measures the proportion of correctly classified instances out of the total instances. A higher accuracy indicates better performance.

#### Confusion Matrix

The confusion matrix is another important evaluation metric for classification tasks. It is a matrix that summarizes the correct and incorrect classifications made by a classifier. Each row of the matrix represents the instances in a true class, while each column represents the instances in a predicted class.

To compute the confusion matrix, we first need to obtain the model's predictions for the test set:

```python
predictions = model.predict(test_images)
```

The predict function returns the predicted probabilities for each class. We need to convert these probabilities into class labels by selecting the class with the highest probability:

```python
predicted_labels = np.argmax(predictions, axis=1)
```

Now we can compute the confusion matrix using scikit-learn's confusion_matrix function:

```python
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(test_labels, predicted_labels)
```

The confusion matrix can be used to identify the model's strengths and weaknesses in classifying different classes. It helps in understanding the types of misclassifications the model makes and can be useful in fine-tuning the model or designing new architectures.

```py
# Compute the confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
confusion = confusion_matrix(test_labels, predicted_labels)
```

### 5. Experiment with different network architectures and compare their performance

- [Keras Tuner: Hyperparameter tuning](https://keras.io/keras_tuner/)
- [Guide to choosing Hyperparameters](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

#### Goals

- Learn how to experiment with different neural network architectures
- Understand the impact of different architectures on performance

#### Key Concepts

- Hyperparameter tuning
- Keras Tuner

##### First, you need to install the Keras Tuner library

```bash
pip install keras-tuner
```

##### Import the necessary libraries

```python
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
```

#### Perform the Tuning
Define a function that builds and returns a Keras model with a given set of hyperparameters. This function should accept an argument hp of type HyperParameters

```python
def build_model(hp):
    model = keras.Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    
    # Use a variable number of hidden layers and units
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(Dense(units=hp.Int('units_' + str(i), 32, 256, step=32),
                        activation='relu'))
    
    model.add(Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(
                    hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
```

This function defines a model with a variable number of hidden layers and units in each layer, as well as a choice of learning rates.

The `hp.Int` function is used to define a hyperparameter that will be tuned by kerastuner. In this case, the hyperparameter is called `num_layers`, and its value can range from 1 to 4, inclusive.

The for loop is used to add a corresponding number of dense layers to the model. For each layer, the number of units is defined as another hyperparameter called `units_i`, where `i` is the index of the current layer. The number of units can range from 32 to 256, in increments of 32.

The activation parameter is set to `'relu'`, which is a common activation function used in neural networks.

Overall, this code allows kerastuner to search for the best combination of hyperparameters for this neural network model, including the number of hidden layers and units in each layer.

##### Create a tuner instance
 
Here, we'll use the RandomSearch tuner, which randomly selects a combination of hyperparameters

```python
tuner = RandomSearch(
    hypermodel=build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='mnist_tuning',
    project_name='mnist_tuning'
)
```

This code sets up a RandomSearch object for hyperparameter tuning using the Keras Tuner library.

Here is what each argument means:

- **hypermodel**: This is the function that will be used to build and compile the model for each trial.
- **objective**: This is the metric that will be used to evaluate the performance of the models. In this case, it is the validation accuracy.
- **max_trials**: This is the maximum number of hyperparameter combinations that will be tried during the search.
- **executions_per_trial**: This is the number of times each model will be trained and evaluated with different initialization and shuffling of the data. The best model is selected based on the average performance across these executions.
- **directory**: This is the directory where the results of the search will be saved. This includes the models, logs, and best hyperparameters.
- **project_name**: This is the name of the project, which is used to organize the results within the directory.

##### Search for the best hyperparameters by fitting the tuner to the training data

##### Peform the search

```python
tuner.search(train_images, train_labels,
             epochs=10,
             validation_split=0.2)
```
This code is using the search method of a KerasTuner instance to perform a hyperparameter search for the best model configuration.

The `search` method takes several arguments, including the training data `train_images` and `train_labels` that the model will be trained on, the number of epochs for training (`epochs`), and the proportion of the training data to use for validation (`validation_split`).

During the search, the tuner will generate a set of hyperparameters to build a model and evaluate it based on its performance on the validation data. It will then use this information to adjust the next set of hyperparameters to try, repeating the process until it has explored the search space for the given number of trials.

At the end of the search, the tuner will return the best model configuration it found based on the specified objective, which in this case is the validation accuracy (`val_accuracy`).


#####  Retrieve the best model and its hyperparameters:

```python
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters()[0]
```

##### Train the best model on the entire training dataset:

```python
best_model.fit(train_images, train_labels, epochs=10)
```

##### Evaluate the best model on the test dataset:

```python
test_loss, test_acc = best_model.evaluate(test_images, test_labels)
```

#### Testing Different Models

```python
# Try different architectures, for example, add more layers or change the number of neurons
model2 = Sequential()
model2.add(Flatten(input_shape=(28, 28)))
model2.add(Dense(256, activation='relu'))
model2.add(Dense(128, activation='relu'))
model2.add(Dense(10, activation='softmax'))

# Compile and train the new model
model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model2.fit(train_images, train_labels, epochs=10)

# Evaluate the new model's performance on the test set
test_loss2, test_acc2 = model2.evaluate(test_images, test_labels)

# Compare the performance of the two models

print("Model 1 Test Accuracy:", test_acc)
print("Model 2 Test Accuracy:", test_acc2)
```

### 6. Document your findings, including the network architectures you tried, their performance, and any insights you gained from the experiments

#### Goals

- Learn how to analyze and document the results of neural network experiments
- Understand the importance of reporting results in a clear and structured manner

#### Key Concepts

- Analysis of results
- Documentation of findings

In this step, you will analyze the results of the experiments you conducted and document your findings. This may include:

- A brief description of each neural network architecture you tried
- A comparison of their performance (e.g., test accuracy)
- Any insights you gained from the experiments, such as the impact of different layer types or activation functions on performance

Remember to structure your documentation clearly and concisely, as it will help you and others understand the experiments and their outcomes better. This practice is also essential for effective communication of research findings in the machine learning community.
