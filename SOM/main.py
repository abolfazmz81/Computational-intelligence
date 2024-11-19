from sklearn.datasets import load_iris
import pandas as pd

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
