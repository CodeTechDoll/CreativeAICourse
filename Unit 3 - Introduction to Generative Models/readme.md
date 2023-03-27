# Introduction to Generative Models

Generative models are a class of machine learning models that aim to generate new samples from the underlying data distribution. They have been used to produce impressive results in various fields, such as image synthesis, text generation, and audio synthesis. In this unit, we will explore some popular generative models, including Autoencoders, Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Transformer-based models.

## Autoencoders

Autoencoders are a type of unsupervised neural network used for data compression and reconstruction. They consist of two main parts: an encoder and a decoder. The encoder maps the input data to a lower-dimensional latent space, while the decoder maps the latent space back to the original data space. The goal of an autoencoder is to learn a compact representation of the input data and be able to reconstruct it with minimal loss.

### Goals

- Understand the concept and architecture of autoencoders
- Learn how to use autoencoders for data compression and reconstruction

### Core Concepts

- Autoencoder architecture (encoder and decoder)
- Latent space representation
- Reconstruction loss

### Resources

- [Autoencoder (Wikipedia)](https://en.wikipedia.org/wiki/Autoencoder)
- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)

### Real World Example

- Image denoising using autoencoders

### Encoder

The encoder is a neural network that takes the input data and compresses it into a lower-dimensional latent space representation. The architecture of the encoder can vary, including fully connected layers, convolutional layers, or recurrent layers, depending on the input data's nature.

### Decoder

The decoder is another neural network that takes the latent space representation and reconstructs the input data. The decoder's architecture should be designed to mirror the encoder's architecture, using layers like fully connected, deconvolutional, or recurrent layers to reconstruct the original input data.

### Loss Function

The loss function for autoencoders is usually based on the reconstruction error between the original input data and the reconstructed data. Common loss functions include mean squared error (MSE) and binary cross-entropy.

### Example: Simple Autoencoder for MNIST

```python
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

# Define the encoder architecture
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)

# Define the decoder architecture
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# Create the autoencoder model
autoencoder = Model(input_img, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder on the MNIST dataset
autoencoder.fit(train_images, train_images,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(test_images, test_images))
```

## Variational Autoencoders (VAEs)

### Goals

- Understand the concept and architecture of variational autoencoders
- Learn how to use VAEs for generating new samples

### Core Concepts

- Variational autoencoder architecture
- Latent variable modeling
- KL divergence

### Resources

- [Variational Autoencoder (Wikipedia)](https://en.wikipedia.org/wiki/Autoencoder#Variational_autoencoder_(VAE))
- [Variational Autoencoders Explained](https://towardsdatascience.com/variational-autoencoders-explained-f5808e8f55d1)

### Real World Example

- Generating new images of handwritten digits using a VAE

Variational Autoencoders (VAEs) are an extension of autoencoders that introduce a probabilistic layer in the latent space. This layer allows VAEs to generate new samples by sampling from the latent space's probability distribution. The VAE architecture consists of an encoder, a latent space, and a decoder.
Encoder

The encoder in a VAE maps the input data to two vectors: one representing the mean and the other representing the standard deviation of a probability distribution in the latent space. This mapping is done using neural networks with various layer types, depending on the input data's nature.
Latent Space

The latent space in a VAE is a probability distribution, usually a Gaussian distribution, with the mean and standard deviation learned by the encoder. To generate a sample in the latent space, we sample from this distribution using the reparameterization trick.
Decoder

The decoder in a VAE takes samples from the latent space and reconstructs the input data. The architecture of the decoder should mirror the encoder's architecture and can include layers like fully connected, deconvolutional, or recurrent layers.
Loss Function

The loss function for VAEs consists of two parts: the reconstruction loss and the KL divergence. The reconstruction loss measures the difference between the original input data and the reconstructed data, while the KL divergence measures the difference between the learned latent space distribution and a prior distribution (usually a standard Gaussian distribution).

#### Example: Variational Autoencoder for MNIST

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

# Define the encoder architecture
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
z_mean = Dense(64)(encoded)
z_log_var = Dense(64)(encoded)

# Sampling function for the latent space
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(64,))([z_mean, z_log_var])

# Define the decoder architecture
decoded = Dense(128, activation='relu')(z)
decoded = Dense(784, activation='sigmoid')(decoded)

# Create the VAE model
vae = Model(input_img, decoded)

# Define the VAE loss function
reconstruction_loss = binary_crossentropy(input_img, decoded) * 784
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)

# Compile the model
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Train the VAE on the MNIST dataset
vae.fit(train_images, None,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_data=(test_images, None))
```

## Generative Adversarial Networks (GANs)

### Goals

- Understand the concept and architecture of generative adversarial networks
- Learn how to use GANs for generating new samples

### Core Concepts

- GAN architecture (generator and discriminator)
- Adversarial training
- GAN loss functions

### Resources

- [Generative Adversarial Networks (Wikipedia)](https://en.wikipedia.org/wiki/Generative_adversarial_network)
- [GANs from Scratch](https://towardsdatascience.com/generative-adversarial-networks-gans-from-scratch-6a8b7d6015b9)

### Real World Example

- Generating new images of faces using a GAN

Generative Adversarial Networks (GANs) are a class of generative models that use two neural networks, a generator, and a discriminator, to generate new samples. The generator creates fake samples, while the discriminator tries to distinguish between fake and real samples. The two networks are trained simultaneously, with the generator learning to create more realistic samples and the discriminator learning to better distinguish between real and fake samples.
Generator

The generator is a neural network that takes a random noise vector as input and generates a fake sample. The architecture of the generator can include various layer types, such as fully connected, deconvolutional, or recurrent layers, depending on the desired output data's nature.
Discriminator

The discriminator is a neural network that takes a sample (either real or generated by the generator) and tries to determine if it's real or fake. The discriminator's architecture can include various layer types, such as fully connected, convolutional, or recurrent layers, depending on the input data's nature.
Loss Function

The loss function for GANs consists of two parts: the generator loss and the discriminator loss. The generator loss measures how well the generator can fool the discriminator, while the discriminator loss measures how well the discriminator can distinguish between real and fake samples.
Example: Simple GAN for MNIST

```python
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Dropout, Input, Reshape, Conv2DTranspose, Flatten
from keras.optimizers import Adam
from keras.datasets import mnist

# Define the generator architecture
generator = Sequential()
generator.add(Dense(128, activation='relu', input_dim=100))
generator.add(Dense(784, activation='sigmoid'))
generator.add(Reshape((28, 28, 1)))

# Define the discriminator architecture
discriminator = Sequential()
discriminator.add(Flatten(input_shape=(28, 28, 1)))
discriminator.add(Dense(128, activation='relu'))
discriminator.add(Dropout(0.5))
discriminator.add(Dense(1, activation='sigmoid'))

# Compile the discriminator
discriminator.compile(optimizer=Adam(lr=0.0002), loss='binary_crossentropy', metrics=['accuracy'])

# Create the GAN model
discriminator.trainable = False
gan_input = Input(shape=(100,))
fake_sample = generator(gan_input)
gan_output = discriminator(fake_sample)
gan = Model(gan_input, gan_output)

# Compile the GAN
gan.compile(optimizer=Adam(lr=0.0002), loss='binary_crossentropy')

# Function to train the GAN
def train_gan(epochs, batch_size):
    (train_images, _), (_, _) = mnist.load_data()
    train_images = train_images / 255.0
    train_images = np.expand_dims(train_images, axis=-1)

    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train the discriminator
        idx = np.random.randint(0, train_images.shape[0], batch_size)
        real_samples = train_images[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_samples = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real_labels)

        # Print the progress
        print(f'Epoch: {epoch + 1}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}')

# Train the GAN on the MNIST dataset
train_gan(epochs=100, batch_size=128)
```

## Transformer-based Models

### Goals

- Understand the concept of transformers and their application in generative tasks
- Learn how to use pre-trained transformer-based models for text generation

### Core Concepts

- Transformer architecture
- Attention mechanisms
- Fine-tuning pre-trained models

### Resources

- [The Illustrated Transformer (Jay Alammar)](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Transformers Library](https://huggingface.co/transformers/)

### Real World Example

- Text generation using GPT-2 or GPT-3

Transformer models are a class of deep learning models that have revolutionized natural language processing (NLP) tasks. They are based on the self-attention mechanism, which allows the model to weigh the importance of different input tokens when making predictions. Transformer models have been used in various NLP tasks, such as machine translation, text summarization, and sentiment analysis.

#### Self-attention

The self-attention mechanism computes a weighted sum of input tokens, with the weights determined by the similarity between the input tokens. This allows the model to focus on the most relevant input tokens when making predictions.

#### Multi-head attention

Multi-head attention is an extension of self-attention that allows the model to compute multiple self-attention scores for each input token. This can help the model capture different types of relationships between the input tokens.

#### Positional encoding

Positional encoding is a method used in transformer models to inject information about the position of tokens in the input sequence. This is done by adding a positional encoding vector to the input token embeddings before feeding them into the self-attention mechanism.
Example: Text Classification with BERT

**BERT (Bidirectional Encoder Representations from Transformers)** is a popular transformer-based model for various NLP tasks. Here is an example of using BERT for text classification:

```python
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

# Load a pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Prepare the dataset
text_data = ['Example sentence 1', 'Example sentence 2', 'Example sentence 3']
labels = [0, 1, 0]  # Binary labels for the text_data

# Tokenize the text data
input_ids = []
attention_masks = []

for text in text_data:
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = tf.concat(input_ids, axis=0)
attention_masks = tf.concat(attention_masks, axis=0)
labels = tf.convert_to_tensor(labels)

# Split the data into train and test sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    input_ids, labels, random_state=42, test_size=0.1
)
train_masks, test_masks, _, _ = train_test_split(
    attention_masks, labels, random_state=42, test_size=0.1
)

# Compile the BERT model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the BERT model on the dataset
history = model.fit(
    [train_inputs, train_masks],
    train_labels,
    epochs=4,
    batch_size=8,
    validation_data=([test_inputs, test_masks], test_labels)
)
```

This example demonstrates how to fine-tune a pre-trained BERT model for a text classification task using the Transformers library. The text_data and labels are placeholders for your dataset. Make sure to replace them with your dataset for the specific problem you are trying to solve.
