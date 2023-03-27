# Unit 2: Introduction to Deep Learning Fundamentals

In this lesson, we will learn about the basics of deep learning, including **neural networks**, **backpropagation**, and various types of **neural network architectures**. We'll also take a look at a real-world example using a pre-trained **Convolutional Neural Network (CNN)** for image classification. By the end of this lesson, you should have a basic understanding of deep learning concepts, even if you have very little coding experience.

## What is Deep Learning?

**Deep learning** is a subset of machine learning that focuses on algorithms inspired by the structure and function of the human brain, called artificial neural networks. These algorithms can automatically learn to represent data by training on large amounts of labeled data, such as images or text. Deep learning has gained popularity in recent years due to its ability to achieve state-of-the-art results on various tasks, such as image classification, natural language processing, speech recognition, and reinforcement learning.

## Artificial Neural Networks

**Artificial neural networks** are a type of machine learning model that mimics the structure and function of biological neurons in the brain. They consist of interconnected layers of artificial neurons or nodes, which process and transform input data to produce the desired output. Neural networks can learn complex patterns and representations from large datasets, making them suitable for various tasks, such as image classification, natural language processing, and reinforcement learning.

A typical neural network consists of the following layers:

- **Input layer**: The input layer receives the input data and passes it to the next layer.
- **Hidden layers**: These layers, positioned between the input and output layers, apply transformations to the input data, such as applying activation functions and adjusting weights and biases.
- **Output layer**: The output layer produces the final output, such as class probabilities or predicted values.

## Backpropagation

**Backpropagation** is a supervised learning algorithm used to train neural networks. It minimizes the error between the predicted output and the actual output (target) by adjusting the weights and biases of the network. Backpropagation calculates the gradient of the loss function with respect to each weight and bias by applying the chain rule, computing the gradient one layer at a time, iterating backward from the last layer to avoid redundant calculations of intermediate terms in the chain rule.

## Convolutional Neural Networks (CNNs)

**Convolutional Neural Networks (CNNs)** are a type of neural network architecture designed to process grid-like data, such as images. They are especially good at capturing spatial and temporal dependencies in data by applying filters (convolutional layers) that learn local patterns. A typical CNN consists of the following layers:

- **Convolutional layer**: Applies a set of filters to the input data, producing feature maps that capture local patterns.
- **Activation function**: Applies a non-linear function (e.g., ReLU) to the output of the convolutional layer, adding non-linearity to the network.
- **Pooling layer**: Reduces the spatial dimensions of the feature maps, making the network more computationally efficient and invariant to small translations.
- **Fully connected layer**: Flattens the feature maps and connects them to the output layer, producing the final output (e.g., class probabilities).

## Real-World Example: Image Classification with a Pre-Trained CNN

In this example, we'll use a pre-trained CNN called ResNet-50 to classify an image. The model has been trained on the ImageNet dataset, which contains millions of images labeled with thousands of object categories. By using the pre-trained model, we can recognize objects in new images without having to train our own model from scratch.

1. Load a pre-trained model: First, we'll use a pre-trained model called ResNet-50. This model has already been trained on millions of images, so it's ready to classify new images without any additional training.
2. Load and preprocess the image: Next, we'll load the image we want to classify and preprocess it to make it compatible with the ResNet-50 model. This includes resizing the image to the appropriate dimensions (e.g., 224 x 224 pixels) and normalizing the pixel values.
3. Make a prediction: We'll input the preprocessed image into the ResNet-50 model, which will analyze the image and output a probability distribution over the possible object categories. This distribution represents the model's confidence in each category.
4. Decode the prediction: To make sense of the model's prediction, we'll convert the probability distribution into a more readable format. This involves selecting the category with the highest probability and mapping it to a human-readable label, such as "dog" or "car".
5. Display the result: Finally, we'll display the predicted class label and its probability. This will tell us what the model thinks is the most likely object in the image.

By following these steps, we can use a pre-trained CNN like ResNet-50 to classify objects in new images without having to build and train our own deep learning model.

Here is the code:

```py
# Import necessary libraries
import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

# Load the pre-trained ResNet-50 model
model = resnet50.ResNet50(weights='imagenet')

# Specify the path to the image file
img_path = 'path/to/image.jpg'

# Load the image and resize it to 224x224 pixels (required by ResNet-50)
img = image.load_img(img_path, target_size=(224, 224))

# Convert the image to a numerical array
x = image.img_to_array(img)

# Add an extra dimension to the array (required by the model)
x = np.expand_dims(x, axis=0)

# Preprocess the image array to be compatible with ResNet-50
x = resnet50.preprocess_input(x)

# Use the model to predict the class of the image
predictions = model.predict(x)

# Decode the predictions to get the human-readable class label
predicted_class = resnet50.decode_predictions(predictions, top=1)[0][0]

# Print the predicted class and its probability
print('Predicted class:', predicted_class[1], ', probability:', predicted_class[2])
```
