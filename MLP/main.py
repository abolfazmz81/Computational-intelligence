import tensorflow as tf
import keras
from keras.api.datasets import mnist
from keras import layers,models

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the images
train_images = train_images.reshape((train_images.shape[0], 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28 * 28)).astype('float32') / 255

# Preprocess the labels (One-hot encoding)
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

# Initialize the model
model = models.Sequential()

# Add the first hidden layer
model.add(layers.Dense(128,activation="relu",input_shape=(784,)))

# Add the second hidden layer
model.add(layers.Dense(64,activation="relu"))

# Add the output layer
model.add(layers.Dense(10,activation="softmax"))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 20 epochs and store history
history = model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_data=(test_images, test_labels))

