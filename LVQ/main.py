import pandas as pd
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

