import numpy as np


class LVQ:

    def __init__(self, prototypes_per_class=1, epoch=20, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.prototypes_per_class = prototypes_per_class
        self.weights = None
        self.labels = None

    def euclidean_distance(self, x, prototype):
        """Calculate Euclidean distance between a sample and a prototype."""
        return np.sqrt(np.sum((x - prototype) ** 2))

    def weight_vectors(self, x, y):
        # Getting all the class types we have
        classes = np.unique(y)
        # Initialize weights and labels as array
        self.weights = []
        self.labels = []
        for cls in classes:
            # Extract x belong to class y
            samples = x[y == cls]
            # Select prototypes_per_class vectors
            selected = samples[
                np.random.choice(samples.shape[0], self.prototypes_per_class, replace=False)
            ]
            self.weights.extend(selected)
            self.labels.extend([cls] * self.prototypes_per_class)

        self.weights = np.array(self.weights)
        self.labels = np.array(self.labels)

    def train(self, x, y):
        # train the LVQ model
        x = np.asarray(x)
        y = np.asarray(y)
        # Initialize weight vectors
        self.weight_vectors(x, y)

        # Loop for each epoch
        for ep in range(self.epoch):
            # Loop all the dataset
            for i, X in enumerate(x):
                # Determining closest prototype
                distances = np.array([self.euclidean_distance(X,l) for l in self.labels])
                closest_index = np.argmax(distances)
                # Check its correct or incorrect
                if y[i] == self.labels[closest_index]:
                    # Pull closer if correct
                    self.weights[closest_index] += self.learning_rate*(X-self.weights[closest_index])
                else:
                    # Push further if incorrect
                    self.weights[closest_index] -= self.learning_rate*(X-self.weights[closest_index])

    def predic(self,x):
        x = np.asarray(x)
        predictions = []
        for X in x:
            distances = np.array([self.euclidean_distance(X, p) for p in self.weights])
            closest_idx = np.argmin(distances)
            predictions.append(self.labels[closest_idx])
        return np.array(predictions)
