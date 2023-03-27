from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras_tuner import RandomSearch
from sklearn.metrics import confusion_matrix
import numpy as np


def main():
    print('Untuned model:')
    untuned_model()
    print('Tuned model:')
    tuned_model()
    
def untuned_model():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the data
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # Define the model
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28))) # Flatten the input images
    model.add(Dense(128, activation='relu')) # Add a hidden layer with 128 neurons and ReLU activation
    model.add(Dense(10, activation='softmax')) # Add an output layer with 10 neurons and a softmax activation

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=10)

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

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    # Evaluate the new model's performance on the test set
    test_loss2, test_acc2 = model2.evaluate(test_images, test_labels)

    # Compare the performance of the two models
    print("Model 1 Test Accuracy:", test_acc)
    print("Model 2 Test Accuracy:", test_acc2)

    # Compute the confusion matrix
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    confusion_matrix(test_labels, predicted_labels)
    
def tuned_model():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the data
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # Create tuner instance
    tuner = RandomSearch(
        hypermodel=build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='mnist_tuning',
        project_name='mnist_tuning')
    
    # Search for best hyperparameters
    tuner.search(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    
    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.fit(train_images, train_labels, epochs=10)
    test_loss_model, test_acc_model = best_model.evaluate(test_images, test_labels)

    
    # Get the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    
    # Build the model with the best hyperparameters
    best_hp_model = build_model(best_hyperparameters)
    best_hp_model.fit(train_images, train_labels, epochs=10)
    test_loss_hp, test_acc_hp = best_hp_model.evaluate(test_images, test_labels)

    print('Best hyperparameters model accuracy:', test_acc_hp)
    print('Best model accuracy:', test_acc_model)
    
    
def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    
    # Add a variable number of hidden layers using Relu activation
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu'))
    
    # Add a final layer with 10 neurons and softmax activation
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


if __name__ == "__main__":
    main()