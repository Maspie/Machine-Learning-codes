import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load your dataset
data = pd.read_csv("./your_data.csv")  

col1, col2 = 'Column1', 'Column2'  

# Preparing the data for clustering
cluster_data = data[[col1, col2]]

# Elbow Method: Find optimal number of clusters
wcss = []
for i in range(1, 11):  # Trying 1 to 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(cluster_data)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Choose the number of clusters based on the elbow graph
num_clusters = 5  

# Applying KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(cluster_data)

# Visualizing the clusters
plt.scatter(cluster_data.iloc[clusters == 0, 0], cluster_data.iloc[clusters == 0, 1], label='Cluster 1')
plt.scatter(cluster_data.iloc[clusters == 1, 0], cluster_data.iloc[clusters == 1, 1], label='Cluster 2')
# Repeat the above two lines for the number of clusters you have
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, color='red', label='Centroids')
plt.title('Clusters of Data Points')
plt.xlabel(col1)
plt.ylabel(col2)
plt.legend()
plt.show()
