# Unit 2: Deep Learning Fundamentals

In this unit, we will cover the fundamentals of deep learning, including neural networks, backpropagation, and various types of neural network architectures, such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU).

## Deep Learning

Deep learning is a subfield of machine learning that focuses on algorithms inspired by the structure and function of the human brain, called artificial neural networks. These algorithms can automatically learn to represent data by training on large amounts of labeled data, such as images or text. Deep learning has gained popularity in recent years due to its ability to achieve state-of-the-art results on various tasks, such as image classification, natural language processing, speech recognition, and reinforcement learning.

In deep learning, artificial neural networks consist of multiple layers of interconnected neurons, which allow the networks to automatically learn hierarchical representations of the data. This hierarchical learning enables deep learning models to capture complex, non-linear relationships between inputs and outputs, making them well-suited for a wide range of tasks.

One real-world example of deep learning is the use of Convolutional Neural Networks (CNNs) for image classification. CNNs are designed to process grid-like data, such as images, and they use convolutional layers to exploit spatial locality and hierarchies in the data. For instance, a CNN can be trained on a large dataset of labeled images to recognize objects in new, unseen images. By learning hierarchical features in the images, such as edges, textures, and shapes, a CNN can achieve high accuracy in object recognition tasks.

Here's a simple example of using a pre-trained CNN for image classification:

```python
import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

# Load a pre-trained ResNet-50 model
model = resnet50.ResNet50(weights='imagenet')

# Load an image and preprocess it for the model
img_path = 'path/to/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = resnet50.preprocess_input(x)

# Use the model to predict the class of the image
predictions = model.predict(x)
predicted_class = resnet50.decode_predictions(predictions, top=1)[0][0]

print('Predicted class:', predicted_class[1], ', probability:', predicted_class[2])
```

In this example, we use the pre-trained ResNet-50 model to classify an image. The model has been trained on the ImageNet dataset, which contains millions of images labeled with thousands of object categories. By using the pre-trained model, we can recognize objects in new images without having to train our own model from scratch.

## Neural Networks and Backpropagation

### Neural Networks

Artificial neural networks are a type of machine learning model inspired by the structure and function of biological neurons in the brain. They consist of interconnected layers of artificial neurons or nodes, which process and transform input data to produce the desired output. Neural networks can learn complex patterns and representations from large datasets, making them suitable for various tasks, such as image classification, natural language processing, and reinforcement learning.

A typical neural network consists of the following layers:

- Input layer: The input layer receives the input data and passes it to the next layer.
- Hidden layers: These layers, positioned between the input and output layers, apply transformations to the input data, such as applying activation functions and adjusting weights and biases.
- Output layer: The output layer produces the final output, such as class probabilities or predicted values.

### Backpropagation

Backpropagation is a supervised learning algorithm used to train neural networks. It minimizes the error between the predicted output and the actual output (target) by adjusting the weights and biases of the network. Backpropagation calculates the gradient of the loss function with respect to each weight and bias by applying the chain rule, computing the gradient one layer at a time, iterating backward from the last layer to avoid redundant calculations of intermediate terms in the chain rule.

### Sourcess

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Online](http://www.deeplearningbook.org/) (especially Chapters 6 and 7)
2. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0). *Nature*, 323(6088), 533-536.
3. [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

## Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a type of neural network architecture designed to process grid-like data, such as images. They are especially good at capturing spatial and temporal dependencies in data by applying filters (convolutional layers) that learn local patterns. A typical CNN consists of the following layers:

- Convolutional layer: Applies a set of filters to the input data, producing feature maps that capture local patterns.
- Activation function: Applies a non-linear function (e.g., ReLU) to the output of the convolutional layer, adding non-linearity to the network.
- Pooling layer: Reduces the spatial dimensions of the feature maps, making the network more computationally efficient and invariant to small translations.
- Fully connected layer: Flattens the feature maps and connects them to the output layer, producing the final output (e.g., class probabilities).

### Sources

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). *Proceedings of the IEEE*, 86(11), 2278-2324.
2. [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
3. [DeepLearning.AI - Convolutional Neural Networks (Coursera)](https://www.coursera.org/learn/convolutional-neural-networks)

## Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a type of neural network architecture designed to process sequences of data. They are especially good at capturing temporal dependencies in data by maintaining an internal state (hidden state) that can remember information from previous time steps. This makes RNNs suitable for tasks such as time series prediction, natural language processing, and speech recognition.

#### Sources

1. Hochreiter, S., & Schmidhuber, J. (1997). [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf). *Neural computation*, 9(8), 1735-1780.
2. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). [Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/pdf/1406.1078.pdf). *arXiv preprint arXiv:1406.1078*.
3. [DeepLearning.AI - Sequence Models (Coursera)](https://www.coursera.org/learn/nlp-sequence-models)

## Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU)

LSTM and GRU are two types of RNN architectures designed to overcome the vanishing gradient problem, which occurs when the gradients of the loss function with respect to the network weights become very small, making it difficult for the network to learn long-range dependencies. Both LSTM and GRU use gating mechanisms to control the flow of information through the network, allowing them to remember information over longer time periods.

### Long Short-Term Memory (LSTM)

LSTM is an RNN architecture that introduces memory cells and three gating units: input gate, forget gate, and output gate. These gates control the flow of information through the memory cell, enabling the LSTM to remember or forget information selectively. LSTMs have been widely used for various sequence-to-sequence tasks, such as machine translation and speech recognition.

### Gated Recurrent Units (GRU)

GRU is an RNN architecture that is a simplified version of LSTM. It uses two gating units: update gate and reset gate. These gates control the flow of information through the hidden state, making it easier for the GRU to capture long-range dependencies while requiring fewer parameters than LSTM. GRUs have been successfully applied to tasks such as language modeling and sentiment analysis.

### Sources

1. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). [Empirical evaluation of gated recurrent neural networks on sequence modeling](https://arxiv.org/pdf/1412.3555.pdf). *arXiv preprint arXiv:1412.3555*.
2. [DeepLearning.AI - Sequence Models (Coursera)](https://www.coursera.org/learn/nlp-sequence-models) (especially the lessons on LSTM and GRU)

## Homework: Build a simple neural network to classify handwritten digits using the MNIST dataset

In this homework assignment, you will build a simple neural network to classify handwritten digits using the MNIST dataset. You will experiment with different network architectures and document your findings.

- Load the MNIST dataset and preprocess the data (normalize, split into train and test sets)
- Design a neural network architecture (e.g., number of layers, types of layers, activation functions)
- Train the neural network using backpropagation and a suitable optimizer (e.g., stochastic gradient descent, Adam)
- Evaluate the performance of the neural network on the test set (e.g., accuracy, confusion matrix)
- Experiment with different network architectures and compare their performance
- Document your findings, including the network architectures you tried, their performance, and any insights you gained from the experiments.

After completing this homework, you should have a better understanding of how neural networks work, how to design and train them, and how to evaluate their performance on a classification task.
