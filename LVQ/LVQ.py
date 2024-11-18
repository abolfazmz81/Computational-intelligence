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
