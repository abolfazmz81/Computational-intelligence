from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
