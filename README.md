# Computational Intelligence Projects

Welcome to the **Computational Intelligence Projects** repository! This collection showcases various projects developed as part of the Computational AI university course, each focusing on different aspects of computational intelligence.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Details](#project-details)
- [Installation and Usage](#installation-and-usage)
- [License](#license)

## Project Overview

This repository contains multiple projects that explore various computational intelligence techniques, including neural networks and genetic algorithms. Each project is organized into its own directory with detailed documentation and source code.

## Project Details

### Project 1: Genetic

**Description:**  
This project provides a solution to the optimal route problem for a package delivery company using the Genetic Algorithm. It also simulates a dynamic city environment, including factors like changing traffic conditions.

**Features:**
- Fully object-oriented implementation, including Car and Package data classes, as well as Distance Matrix and Traffic System classes.
- Dynamically reassigns packages mid-transportation if a more optimal route is available.
- Implements multiple mutation methods.
- Uses Tournament Selection and Single-Point Crossover.
- Applies penalties for general delays, priority package delays, and exceeding capacity limits.

**Directory:** [`./Genetic`](./Genetic)

### Project 2: Kmeans

**Description:**  
This project focuses on analyzing student data and clustering them into five (final K) groups to study their behavior and suggest suitable educational methods.

**Features:**
- Implements methods to handle null values and outlier data.
- Applies One-Hot Encoding to categorical (conceptual) columns.
- Assigns dynamic weights based on areas of interest.
- Determines the optimal K and visualizes clustering results using Elbow Method, Silhouette Score, Davies-Bouldin Index, and Gap Statistics.
- Utilizes Principal Component Analysis (PCA) for dimensionality reduction.

**Directory:** [`./Kmeans`](./Kmeans)

### Project 3: LVQ

**Description:**  
This project implements a cheat detection system for the **AlgoNIT** programming competition using LVQ (Learning Vector Quantization) based on participants' data, with two prototypes: **Cheater** and **Not Cheater**

**Features:**
- Implements LVQ (Learning Vector Quantization) in a separate class for future reuse.
- Uses Euclidean distance as the primary distance metric.
- Filters and removes rows with meaningless or invalid values.
- Applies Min-Max Scaler for data normalization.
- Evaluates model performance using Accuracy Score.

**Directory:** [`./LVQ`](./LVQ)

### Project 4: MLP

**Description:**  
This project implements a classification system for input images trained on the MNIST dataset using a four-layer **MLP (Multilayer Perceptron)**. The architecture consists of an input layer with **784 neurons**, two hidden layers with **128 neurons** and **64 neurons**, and an output layer with **10 neurons** for digit classification.

**Features:**
- Inverts the colors of an input grayscale image.
- Preprocesses all images in a specified folder, allowing batch testing of multiple images.
- Uses **ReLU** activation for hidden layers and **Softmax** activation for the output layer.
- Optimized with **Adam** and trained using **Cross-Entropy** Loss.
- Trains for 20 epochs and evaluates performance using accuracy.
- Visualizes training progress by plotting:
   - Accuracy Graph: Training Accuracy vs. Validation Accuracy.
   - Loss Graph: Training Loss vs. Validation Loss.
- Performs a final test using custom handwritten digits.

**Directory:** [`./MLP`](./MLP)

### Project 5: Reinforcement Learning

**Description:**  
This project implements a recommendation system for an online movie streaming platform using a **Q-Learning agent** and **reinforcement learning**. It suggests movies based on a user's previous interactions and interests. Initially, it generates a random recommendation and predicts user interactions based on genre preferences. Over time, it dynamically adjusts recommendations, ultimately providing a more personalized movie suggestion.

**Features:**
- Implements a complete classification by separating the Q-agent, data generator, and random initialization recommendation classes.
- Modular Q-agent class, designed for future reuse.
- Uses Faker to generate 100 users and 50 content items, exporting the data as a CSV file.
- Simulates dynamic user interactions based on the match between movie genres and users' favorite genres.
- Trains for 100 epochs with an exploration rate of 0.2.
- After training, recommends movies for 5 random users.

**Directory:** [`./Reinforcement Learning`](./Reinforcement%20Learning)

### Project 6: SOM

**Description:**  
This project implements a clustering system for the Iris dataset using a **Self-Organizing Map (SOM)** model with a 5×5 grid, provided by the **MiniSom** package. It visualizes the clustering results using a scatter plot.

**Features:**
- Checks for null values and provides a descriptive analysis of the dataset.
- Normalizes selected numerical columns using **Min-Max Scaler**, scaling values between 0 and 1.
- Implements a **5×5 grid** with a **0.1 learning rate**.
- Uses a radius of 2.5 and trains for 500 iterations.
- Maps neurons to their corresponding labels for better interpretability.
- Evaluates performance using Accuracy.
- Computes and visualizes a **Confusion Matrix** for better result interpretation

**Directory:** [`./SOM`](./SOM)

## Installation and Usage

To run any of these projects locally:

1.**Clone the repository:**
   ```bash
   git clone https://github.com/abolfazmz81/Computational-intelligence.git
   cd Computational-intelligence
   ```

2.**Navigate to the project directory:**
  ```bash
  cd [ProjectDirectory]
  ```
Replace [ProjectDirectory] with the specific project folder.

3.**Install dependencies:** 
Ensure you have Python installed. Install the required packages using:
  ```bash
  pip install -r requirements.txt
  ```
Each project directory contains its own requirements.txt file with the necessary dependencies(except knn).

4.**Run the project:** 
Execute the main script for the project:
  ```bash
  python main.py
  ```

## License
This repository is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more information.

