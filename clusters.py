from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import argparse
import sys

parser=argparse.ArgumentParser(prog=sys.argv[0],description="Eval parser")
parser.add_argument("--seed",default =0, type=int,help="flag to use barlow twins backbone to initialise weight")
args=parser.parse_args()

seed=args.seed
np.random.seed(args.seed)
# Assuming you have a DataFrame `df` with your data, and 'labels' column is not used in PCA
df= pd.read_csv('Dataset_features.csv')
df_features=df.drop(columns=['mask_path'])
df_features.fillna(0)
df_features.replace([np.inf, -np.inf], 0, inplace=True)

# Standardize the features to have mean=0 and variance=1
features = StandardScaler().fit_transform(df_features)

#check for and replace nan
print(np.isinf(features).sum())
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# Apply PCA, while preserving 95% of the variance
pca = PCA(n_components=0.99)
features_pca = pca.fit_transform(features)

# Set 'k' to the number of principal components
k = features_pca.shape[1]
print('k=',k)

range_n_clusters=np.arange(2,k)


# Initialize variables to store the best score and labels
best_score = -1  # Start with -1 as Silhouette Score ranges from -1 to 1
best_n_clusters = 0
best_labels = None

for n_clusters in range_n_clusters:
    # Initialize KMeans with n_clusters
    kmeans = KMeans(n_clusters=n_clusters)

# Perform K-means clustering
kmeans = KMeans(n_clusters=k)
kmeans.fit(features_pca)

# The labels_ attribute gives the list of clusters each sample belongs to
clusters = kmeans.labels_

# Create a new DataFrame with indices, original labels and assigned clusters
df_clusters = df[['mask_path']].copy()  # create a copy of the labels column
df_clusters['cluster'] = clusters  # add the cluster assignments

cluster_1_files= df_clusters[df_clusters['cluster'] == 1]

from scipy.spatial import distance

# closest_points = []
# for i in range(kmeans.n_clusters):
    # cluster_points_indices = np.where(kmeans.labels_ == i)[0]  # get global indices of all points in cluster i
    # cluster_points = features_pca[cluster_points_indices]  # get all points in cluster i
    # cluster_center = kmeans.cluster_centers_[i]  # get the center of cluster i
    # distances = [distance.euclidean(p, cluster_center) for p in cluster_points]  # calculate distances
    # closest_point_idx_local = np.argmin(distances)  # get the local index of the closest point
    # closest_point_idx_global = cluster_points_indices[closest_point_idx_local]  # convert to global index
    # closest_points.append(closest_point_idx_global)
# closest_points=np.unique(np.array(closest_points))
# print(len(closest_points),closest_points)
# ensemble_files=df['mask_path'].iloc[closest_points].str.slice(0,-26)
# ensemble_files.to_csv(f'selected_files_seed{seed}.csv')


import pandas as pd

# Create a Pandas Excel writer using XlsxWriter as the engine
writer = pd.ExcelWriter('selected_files_seed.xlsx', engine='xlsxwriter')

for i in range(kmeans.n_clusters):
    cluster_points_indices = np.where(kmeans.labels_ == i)[0]  # get global indices of all points in cluster i
    cluster_points = features_pca[cluster_points_indices]  # get all points in cluster i
    cluster_center = kmeans.cluster_centers_[i]  # get the center of cluster i
    distances = [distance.euclidean(p, cluster_center) for p in cluster_points]  # calculate distances
    filepath=df['mask_path'].iloc[cluster_points_indices].str.slice(0,-26)
    # Create a DataFrame
    df_sep = pd.DataFrame({
        'Distance': distances,
        'Index': filepath
    })

    # Sort DataFrame by distance
    df_sep.sort_values('Distance', inplace=True)

    # Write each DataFrame to a specific sheet
    df_sep.to_excel(writer, sheet_name=f'Cluster_{i}', index=False)

# Save the excel
writer.save()


