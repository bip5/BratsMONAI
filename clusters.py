from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

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

# Define the range of variations
variation_range = np.arange(50, 100, 5)  # 50%, 55%, ..., 95%

# Initialize a dictionary to store silhouette scores
silhouette_scores_dict = {}

for variation in variation_range:
    x = variation / 100  # Convert to a fraction
    print(f"Running PCA for {x*100}% variance")

    # Apply PCA, while preserving x% of the variance
    pca = PCA(n_components=x)
    features_pca = pca.fit_transform(features)

    # Set 'k' to the number of principal components
    k = features_pca.shape[1]
    print(f'Number of principal components (k) = {k}')

    range_n_clusters = np.arange(2, k + 1)

    # Initialize a list to store silhouette scores for this variation level
    silhouette_scores_list = []

    for n_clusters in range_n_clusters:
        # Initialize KMeans with n_clusters
        kmeans = KMeans(n_clusters=n_clusters)
        
        # Fit the model and get the cluster labels
        cluster_labels = kmeans.fit_predict(features_pca)

        # Calculate the Silhouette Score
        silhouette_avg = silhouette_score(features_pca, cluster_labels)
        print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

        # Append the silhouette score to the list
        silhouette_scores_list.append(silhouette_avg)

    # Store the list of silhouette scores in the dictionary
    silhouette_scores_dict[variation] = silhouette_scores_list
    

# Initialize an empty DataFrame to store silhouette scores
silhouette_scores_df = pd.DataFrame()

# Populate the DataFrame
for variation, scores in silhouette_scores_dict.items():
    # Create a temporary Series with the silhouette scores
    temp_series = pd.Series(scores, name=variation)
    
    # Append the Series as a new column in the DataFrame
    silhouette_scores_df = pd.concat([silhouette_scores_df, temp_series], axis=1)

# Rename the columns to indicate the levels of variation
silhouette_scores_df.columns = [f"{col}% Variation" for col in silhouette_scores_df.columns]

# Fill NaN values for missing scores
silhouette_scores_df.fillna(value=np.nan, inplace=True)
# Now, silhouette_scores_df contains the silhouette scores with NaNs for missing values
silhouette_scores_df.index = silhouette_scores_df.index + 2
silhouette_scores_df.index.name = 'Number of Clusters'
# print(silhouette_scores_df)
silhouette_scores_df.to_csv('silhouette_scores.csv')
        # if silhouette_avg > best_score:
            # best_score = silhouette_avg
            # best_n_clusters = n_clusters
            # best_labels = cluster_labels
        

# Create the line plot
plt.figure(figsize=(12, 8))

# Loop through each column (variation level) in the DataFrame
for col in silhouette_scores_df.columns:
    plt.plot(silhouette_scores_df.index, silhouette_scores_df[col], label=col)

# Add title and labels
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

# Add a legend
plt.legend(title='Variation Level')
plt.grid(True)

# Add custom ticks at each cluster center
plt.xticks(silhouette_scores_df.index)

# Add faint dashed lines at each cluster center
for x in silhouette_scores_df.index:
    plt.axvline(x=x, linestyle='--', linewidth=0.5, color='grey')
# Show the plot
plt.savefig('silhouette_scores.jpg')

sys.exit()  
    



from scipy.spatial import distance


import pandas as pd

# Create a Pandas Excel writer using XlsxWriter as the engine
writer = pd.ExcelWriter('selected_files_seed.xlsx', engine='xlsxwriter')

for i in range(best_n_clusters):
    cluster_points_indices = np.where(best_labels == i)[0]  # get global indices of all points in cluster i
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


