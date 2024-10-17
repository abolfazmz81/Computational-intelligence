import keras
from keras.api.datasets import mnist
from keras import layers,models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Function to preprocess an external(handwritten) image
def preprocess_image(image_path):
    # Load the image and convert to grayscale
    img = Image.open(image_path).convert('L')  # 'L' converts to grayscale

    # Resize to 28x28 pixels (same as MNIST)
    img = img.resize((28, 28))

    # Convert image to a NumPy array and normalize (0-255 -> 0-1)
    img = np.array(img) / 255.0

    # Flatten the image to a 784-element vector (just like the MNIST dataset)
    img = img.reshape((784,))

    return img

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

# Plot the accuracy and loss over the training period
plt.figure(figsize=(12, 5))

# Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show the plots
plt.show()

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print(f"Test Accuracy: {test_accuracy:.4f}")
