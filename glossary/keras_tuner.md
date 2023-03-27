# Keras Tuner

[Keras Tuner](https://keras.io/keras_tuner/) is a hyperparameter tuning library designed for Keras models. It helps you find the optimal set of hyperparameters for your deep learning models, improving their performance without requiring manual trial and error.

## Key Concepts

- **Hyperparameter**: A parameter whose value is set before the learning process begins. Examples of hyperparameters include learning rate, batch size, and the number of layers and neurons in a neural network.

- **Hyperparameter Tuning**: The process of searching for the best hyperparameter values that maximize the model's performance on a given task.

- **Search Space**: The range of possible hyperparameter values to be considered during the tuning process.

## Using Keras Tuner

### 1. **Install Keras Tuner**

To start using Keras Tuner, you need to install it via pip:

```shell
pip install keras-tuner
```

### 2. **Define the Model**

Create a function that builds and returns the model you want to optimize. In this function, specify the hyperparameters you want to tune as arguments with their search space.

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner import HyperModel

class MyHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                        activation='relu',
                        input_shape=self.input_shape))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
```

### 3. **Instantiate the Tuner**

Choose the tuner you want to use (e.g., `RandomSearch`, `BayesianOptimization`, or `Hyperband`) and provide the necessary arguments, such as the model-building function, the search space, and the evaluation metric.

```py
from kerastuner import RandomSearch

input_shape = (28, 28)
num_classes = 10

hypermodel = MyHyperModel(input_shape, num_classes)

tuner = RandomSearch(hypermodel,
                     objective='val_accuracy',
                     max_trials=10,
                     directory='my_dir',
                     project_name='helloworld')
```

### 4. **Retrieve the Best Model**

After the search is complete, you can get the best model and its hyperparameters using the get_best_models and get_best_hyperparameters methods.

```py
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters found:\n", best_hyperparameters)
```
