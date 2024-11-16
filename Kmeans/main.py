import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from gapstatistics import gapstatistics
from sklearn.decomposition import PCA

# silencing warnings
warnings.filterwarnings('ignore')

# Change display sizes
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)


def handle_nulls(data):
    # Impute missing values with the mean of each column
    for i, column in enumerate(data.columns):
        if data[column].isnull().any():  # Check if there are any null values in the column
            column_mean = round(data[
                                    column].mean())  # Calculate the round mean of columns, all the columns that are mean have Integer values
            print(i)  # Shows above theory
            data[column].fillna(column_mean, inplace=True)  # Fill nulls with the mean floor
    return data


def handle_outliers(df, method):
    for column in df.select_dtypes(include=["float64", "int64"]).columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1  # Interquartile range

        # Define lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if method == "cap":
            # Cap outliers to the lower or upper bound
            df[column] = df[column].apply(
                lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
        elif method == "remove":
            # Remove rows with outliers in this column
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        elif method == "mean":
            # Replace outliers with the mean of the column
            column_mean = df[column].mean()
            df[column] = df[column].apply(lambda x: round(column_mean) if x < lower_bound or x > upper_bound else x)
    return df


# Load the data
data = pd.read_csv('data.csv')

# Change null values
data = handle_nulls(data)

# Cap outliner and noise values
data = handle_outliers(data, "cap")

# Applying one hot encoding
data = pd.get_dummies(data, columns=['StudyEnvironment'], prefix='StudyEnv')

# Select only numerical columns
numerical_columns = data.select_dtypes(include=["float64", "int64"]).columns.drop("StudentID")

# Initialize the scaler
scaler = MinMaxScaler()

# Apply the scaler to the numerical columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
### step 2: custom measure ###
# Weights for custom distance measure
weights = {
    'StudyHours': 3,  # Higher weight for StudyHours
    'MidtermScores': 3,  # Higher weight for MidtermScores
    'ExtracurricularActivities': 3,  # Higher weight for ExtracurricularActivities
    'Attendance': 1,  # Lower weight for other features
    'HomeworkScores': 1,
    'ProjectScores': 1,
    'OnlineResourcesUsage': 1
}

### step 3 & 4: optimal k and KMeans development ###

# Apply weights to the data
for column, weight in weights.items():
    data[column] = data[column] * weight

# drop irrelevant column student id
data = data.drop(columns=["StudentID"])

# K values to test
K_range = range(2, 11)

# Elbow
elbow = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
    elbow.append(kmeans.inertia_)

plt.plot(K_range, elbow, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method")
plt.show()

# Silhouette Score
sh_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    score = silhouette_score(data, kmeans.labels_)
    sh_scores.append(score)

plt.plot(K_range, sh_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Method")
plt.show()

# Davies-Bouldin Index
db_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    score = davies_bouldin_score(data, kmeans.labels_)
    db_scores.append(score)

plt.plot(K_range, db_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Davies-Bouldin Index")
plt.title("Davies-Bouldin Index Method")
plt.show()

# gap statistics
gp = gapstatistics.GapStatistics(return_params=True)
optimum, params = gp.fit_predict(K=10, X=data.__array__())

gp.plot()
### step 5: Cluster analyze ###
# Final K=5
kmeans = KMeans(n_clusters=5, random_state=0).fit(data)

# Add the cluster labels to the DataFrame
data['Cluster'] = kmeans.labels_

# Calculate mean values for each cluster
cluster_summary = data.groupby('Cluster').mean()

# Display the summary statistics for each cluster
print(cluster_summary)

### step 6: Dimension reduction ###

features = data.drop(columns=['Cluster'])  # Avoid including 'Cluster' column

# Step 2: Apply PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 components
data_pca = pca.fit_transform(features)

# Adding back Cluster to dataframe
pca_df = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = data['Cluster']

