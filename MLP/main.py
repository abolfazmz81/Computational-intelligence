import keras
from keras.api.datasets import mnist
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


# Function to preprocess an external(handwritten) image
def preprocess_image(image_path):
    # Load the image and convert to grayscale
    img = Image.open(image_path).convert('L')  # 'L' converts to grayscale

    # Resize to 28x28 pixels (same as MNIST)
    img = img.resize((28, 28))

    # Convert image to a NumPy array
    img = np.array(img)

    # Invert the image (black on white -> white on black)
    img = np.invert(img)

    # Flatten the image to a 784-element vector (just like the MNIST dataset)
    img = img.reshape((784,)).astype('float32') / 255

    return img


# Function to display the image and its predicted label
def display_prediction(image, predicted_label):
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()


# Function to load and preprocess all images in the directory
def load_and_preprocess_images(directory):
    images = []
    image_files = os.listdir(directory)  # List of image filenames in the directory

    for image_file in image_files:
        # Preprocess each image
        image_path = os.path.join(directory, image_file)
        preprocessed_image = preprocess_image(image_path)
        images.append(preprocessed_image)

    # Convert the list of images to a NumPy array (batch)
    images = np.array(images)

    return images, image_files


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
model.add(layers.Dense(128, activation="relu", input_shape=(784,)))

# Add the second hidden layer
model.add(layers.Dense(64, activation="relu"))

# Add the output layer
model.add(layers.Dense(10, activation="softmax"))

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

# Directory where external(handwritten images are stored
image_directory = "./handwritten_numbers"

# Load and preprocess images
external_images, image_filenames = load_and_preprocess_images(image_directory)

# Make predictions for the images
predictions = model.predict(external_images)

# Display the images and the model's predictions
for i, image_filename in enumerate(image_filenames):
    # Reshape the image back to 28x28 for display purposes
    image_reshaped = external_images[i].reshape(28, 28)

    # Get the predicted label
    predicted_label = np.argmax(predictions[i])

    # Display the image and the result
    print(f"Image: {image_filename}, Predicted Label: {predicted_label}")
    display_prediction(image_reshaped, predicted_label)
