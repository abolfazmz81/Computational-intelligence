from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

# Change display sizes
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)

# Load the Iris dataset
iris = load_iris()

# Convert to a DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Display the first rows
print(iris_df.head())

# Check data types
data_types = iris_df.dtypes

# Check for missing values
missing_values = iris_df.isnull().sum()

# Statistics for feature distributions
feature_distributions = iris_df.describe()

# Display results
print("Data Types:\n", data_types)
print("\nMissing Values:\n", missing_values)
print("\nFeature Distributions:\n", feature_distributions)

# Columns to normalize
columns_to_normalize = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

# Initialize the Min Max scaler(range between 0 and 1)
scaler = MinMaxScaler()

# Apply scaler
iris_df[columns_to_normalize] = scaler.fit_transform(iris_df[columns_to_normalize])

# Display normalized data
print(iris_df.head(15))

# Prepare the feature data
X = iris_df.iloc[:, :-1].values  # Features

# Define SOM parameters
som_dimensions = (5, 5)  # 10x10 grid
learning_rate = 0.1
initial_radius = max(som_dimensions) / 2
num_iterations = 500

# Initialize SOM
som = MiniSom(x=som_dimensions[0],
              y=som_dimensions[1],
              input_len=X.shape[1],
              sigma=initial_radius,
              learning_rate=learning_rate)

# Randomly initialize weights
som.random_weights_init(X)

# Train the SOM
# Automatically handles updating the weights and Gradual reduction of learning rate and radius
som.train_random(data=X, num_iteration=num_iterations)

# Allocate each data point to the closest neuron (BMU)
winning_neurons = [som.winner(x) for x in X]

# Convert BMU(Best Matching Unit) coordinates (row, col) into a unique cluster ID
clusters = ['{}-{}'.format(row, col) for row, col in winning_neurons]

# Add the cluster information to the original dataset
iris_df['cluster'] = clusters

# Visualization: Scatter plot for clustering
feature_x = 'sepal length (cm)'
feature_y = 'sepal width (cm)'

# Assign unique colors to each cluster
unique_clusters = iris_df['cluster'].unique()
cluster_colors = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
colors = iris_df['cluster'].map(cluster_colors)

# Scatter plot
plt.figure(figsize=(10, 7))
plt.scatter(
    iris_df[feature_x],
    iris_df[feature_y],
    c=colors,
    cmap='tab10',
    s=50,
    alpha=0.7
)

# Add labels and title
plt.title('SOM Clustering Visualization (Sepal Length vs Sepal Width)')
plt.xlabel(feature_x)
plt.ylabel(feature_y)

# Create a legend
scatter_labels = [f"Cluster {c}" for c in cluster_colors.keys()]
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label=label,
               markersize=10, markerfacecolor=plt.cm.tab10(idx / len(unique_clusters)))
    for label, idx in cluster_colors.items()
], title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()
