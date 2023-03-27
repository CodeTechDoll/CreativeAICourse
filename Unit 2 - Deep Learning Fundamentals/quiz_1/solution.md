# Quiz 1 Solutions

## Theoretical

### 1. What is the primary difference between a Convolutional Neural Network (CNN) and a Recurrent Neural Network (RNN)?

The primary difference between a CNN and an RNN lies in their architecture and the types of problems they are designed to solve. CNNs are designed for processing grid-like data, such as images, and they use convolutional layers to exploit spatial locality and hierarchies in the data. RNNs, on the other hand, are designed for processing sequences of data, such as time series or natural language, and they maintain internal states that can capture information from past elements in the sequence.

- [CNNs - Stanford CS231n](https://cs231n.github.io/convolutional-networks/)
- [RNNs - Stanford CS224n](https://cs224n.stanford.edu/lecture_notes/notes1.pdf)

### 2. Explain the purpose of activation functions in neural networks and provide examples of two commonly used activation functions

Activation functions serve as non-linear transformations applied to the outputs of neurons in a neural network. They introduce non-linearity into the network, allowing it to learn complex, non-linear relationships between input and output data. Two commonly used activation functions are the Rectified Linear Unit (ReLU) and the Sigmoid function. ReLU is defined as `f(x) = max(0, x)` and is commonly used in the hidden layers of neural networks. The Sigmoid function, defined as `f(x) = 1 / (1 + exp(-x))`, is often used in the output layer for binary classification tasks.

- [Activation Function - Deep Learning Book](https://www.deeplearningbook.org/contents/mlp.html)

### 3.  What is the vanishing gradient problem, and how does it affect the training of deep neural networks?

The vanishing gradient problem occurs when gradients of the loss function with respect to the network's parameters become very small as they are backpropagated through the layers, especially in deep networks. This can cause the weights in the early layers of the network to update very slowly, making it difficult for the network to learn the appropriate features from the input data. The vanishing gradient problem is more pronounced when using activation functions with small gradients, such as the sigmoid or hyperbolic tangent (tanh) functions.

- [Visualizing Vanishing Gradient Problem](https://machinelearningmastery.com/visualizing-the-vanishing-gradient-problem/)

### 4. Briefly describe the concept of residual connections introduced by the ResNet architecture and explain how they help with training deeper neural networks

Residual connections, introduced by the ResNet architecture, are a technique for addressing the vanishing gradient problem in deep neural networks. They involve adding "skip connections" that bypass one or more layers, allowing the output of an earlier layer to be added to the output of a later layer. This enables the network to learn a residual function (i.e., the difference between the input and output) rather than learning the output directly. By incorporating residual connections, the gradients can flow more easily through the network, facilitating the training of deeper neural networks.

- [ResNet - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## Coding

### 5. Design a simple neural network using Keras for a binary classification problem. The input data has 20 features. Use the Sequential model with a single hidden layer containing 64 neurons and a ReLU activation function. Include an output layer with the appropriate activation function for binary classification

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
```
