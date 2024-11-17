import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Load the dataset
file_path = 'LVQ_data.csv'
data = pd.read_csv(file_path)

# Remove rows with missing values
data_cleaned = data.dropna()

# Filter out rows with invalid or inconsistent values
data_cleaned = data_cleaned[
    (data_cleaned["TimeTaken"] > 0) &  # Positive TimeTaken
    (data_cleaned["NumberOfAttempts"] > 0) &  # Positive NumberOfAttempts
    (data_cleaned["CodeSimilarity"] >= 0) &  # Non-negative CodeSimilarity
    (data_cleaned["NumberOfRequests"] >= 0)  # Non-negative NumberOfRequests
]

# Display results of cleaning
print(f"Total rows remaining: {data_cleaned.shape[0]}")

# Columns we need to normalize
columns_to_normalize = ["TimeTaken", "NumberOfAttempts", "CodeSimilarity", "NumberOfRequests"]

# Initialize the Min Max scaler(range between 0 and 1)
scaler = MinMaxScaler()

# Apply the scaler to the columns
data_cleaned[columns_to_normalize] = scaler.fit_transform(data_cleaned[columns_to_normalize])

# Display the normalized data
print("Normalized Data:")
print(data_cleaned.head())


