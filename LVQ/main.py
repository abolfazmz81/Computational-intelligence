import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Change display sizes
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)

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

# Drop and display ID column
data_cleaned = data_cleaned.drop(columns=["ParticipantID"])
print(data_cleaned.head())

# Separate features and target
X = data_cleaned.drop(columns=["IsCheater"])  # Features
y = data_cleaned["IsCheater"]  # Target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

