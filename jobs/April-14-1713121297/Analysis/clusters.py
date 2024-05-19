import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler,RobustScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
import seaborn as sns
from datetime import datetime
import os
from Input.dataset import (
test_indices,train_indices,
val_indices
)
from Input.config import raw_features_filename

'''Script to separate extracted features into defined clusters'''

parser = argparse.ArgumentParser(prog=sys.argv[0],description="Eval parser")
parser.add_argument("--seed",default =0, type=int,help="seed")
parser.add_argument("--normaliser",default =2, type=int,help="0=standard scaler,1=SS+Normalizer(datapoint,euc norm),2=MinMaxScaler, 3=RobustScaler")
args=parser.parse_args()




seed=args.seed
np.random.seed(args.seed)

number_clusters=4
# filename='B23allEncoded_maxfeatures'
filename= raw_features_filename#'trainFeatures_01April'#'Encoded_maxfeatures'# 'merged_features'
df= pd.read_csv(f'/scratch/a.bip5/BraTS/{filename}.csv')
# df= pd.read_csv('Dataset_features.csv')
df.drop_duplicates(inplace=True)
# df.sort_values('mask_path',inplace=True)
# print('COLUMNS in ORIGINAL DF', df.columns)
if filename=='merged_features':
    df['mask_path'] = df['mask_path'].map(lambda x: '/'+os.path.join(*x.split('/')[:-1]))

# df.drop(['a_centroid', 's_centroid','c_centroid'], axis=1, inplace=True)
df_features=df.drop(columns=['Unnamed: 0','mask_path','subject_id'],errors='ignore') 

df_features.drop_duplicates(inplace=True)
df_features.fillna(0,inplace=True)
df_features.replace([np.inf, -np.inf], 0, inplace=True)

mask_col='mask_path'#'Unnamed: 0' # 
    
now = datetime.now()
formatted_time =now.strftime('%Y-%m-%d_%H-%M-%S')
plot_dir=os.path.join('./plots',formatted_time)

os.makedirs(plot_dir,exist_ok=True)


def normalise(df_features,normaliser=args.normaliser):
    if normaliser==0:
        # Standardize the features to have mean=0 and variance=1
        features = StandardScaler().fit_transform(df_features) 
    
    if normaliser==1:
        features = StandardScaler().fit_transform(df_features) 
        
        scaler = Normalizer()
        features = scaler.fit_transform(features)

    if normaliser==2:
        scaler = MinMaxScaler()
        features = scaler.fit_transform(df_features)
        
    if normaliser==3: 
        scaler = RobustScaler()
        features = scaler.fit_transform(df_features)
    
    if normaliser==4: 
        features = MinMaxScaler().fit_transform(df_features) 
        
        scaler = Normalizer()
        features = scaler.fit_transform(features)
        
        
        
    return features

    

def sil_plots(df_features):
    

    features=normalise(df_features)
    
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
        fit_with=features_pca
        for n_clusters in range_n_clusters:
            # Initialize KMeans with n_clusters
            kmeans = KMeans(n_clusters=n_clusters)
            
            # Fit the model and get the cluster labels
            cluster_labels = kmeans.fit_predict(fit_with)

            # Calculate the Silhouette Score
            silhouette_avg = silhouette_score(fit_with, cluster_labels)
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
    
    return None


def mahalanobis_distance(points, center):
    # Calculate the covariance matrix
    cov_matrix = np.cov(points, rowvar=False)
    corrcoef=np.corrcoef(points,rowvar=False)
    # Calculate the inverse of the covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    #TURN THIS BACK TO inv from pinv once feature selection complete
    distances = []
    for p in points:
        diff = p - center
        value=diff.T @ inv_cov_matrix @ diff
        print(value)
        if value>0:
            dist = np.sqrt(value)
        else:
            dist=2
        distances.append(dist)
    
    return distances,corrcoef

def plot_correlations(cluster_number, corr_df):
    # Assuming corr_df is the correlation matrix for the cluster
    for col in corr_df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=corr_df.columns, y=corr_df[col])
        plt.xticks(rotation=90)  # Rotate column labels for clarity
        plt.title(f"Correlation of {col} with Other Features - Cluster {cluster_number}")
        plt.ylabel(f"Correlation with {col}")
        plt.tight_layout()  # Adjust layout for better visualization
        plt.savefig(f"{plot_dir}/Cluster_{cluster_number}_Correlation_{col}.png")
        plt.close()
        

def plot_heatmap(cluster_number, corr_df):
    plt.figure(figsize=(12, 12))
    sns.heatmap(corr_df, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"Correlation Heatmap - Cluster {cluster_number}")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Cluster_{cluster_number}_Heatmap.png")
    plt.close()

def plot_pairgrid(cluster_number, subset_df):
    g = sns.PairGrid(subset_df)
    g.map_upper(plt.scatter, s=10)  # Scatter plot for upper grid
    g.map_diag(sns.histplot)  # Histogram for diagonal
    g.map_lower(sns.kdeplot)  # KDE for lower grid
    g.savefig(f"{plot_dir}/Cluster_{cluster_number}_PairGrid.png")

def plot_bubble_correlation(cluster_number, corr_df):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # We'll use a mask to get the upper triangle (as correlation matrices are symmetric)
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    
    for i, j in zip(*np.where(~mask)):
        color = 'red' if corr_df.iloc[i, j] < 0 else 'blue'
        size = abs(corr_df.iloc[i, j]) * 1000  # Scale size for better visualization
        ax.scatter(i, j, s=size, c=color, marker='o', alpha=0.6)
    
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=90)
    ax.set_yticklabels(corr_df.columns)
    plt.title(f"Correlation Bubble Plot - Cluster {cluster_number}")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Cluster_{cluster_number}_BubbleCorrelation.png")
    plt.close()

# Function to count the nearest points to each cluster center
def count_nearest_points(check_data, kmeans,max_bound,min_bound):
    counts = [[],[],[],[]]  # Assuming 4 clusters
    for i,point in enumerate(check_data.iterrows()):
        point=np.array(point[1])
        point_norm=(point-min_bound)/(max_bound-min_bound)
        distances = [np.linalg.norm(point_norm - center) for center in kmeans.cluster_centers_]
        # Find the closest cluster for each point
        closest_cluster = np.argmin(distances)
        min_distance= distances[closest_cluster]
        original_index=val_indices[i]
        counts[closest_cluster].append([original_index,min_distance])
    return counts   
 
 
''' Cluster the training data using kmeans and then check which of the validation points are closest to which cluster using count_nearest_points function. Then save val and train samples together in a excel file'''

def save_cluster_files(df_features, n_clusters=number_clusters,retries=3):
    if retries <= 0:
       return 'Failed to validate clusters after specified attempts. Please modify function signature to increase attempts'
    retries=retries-1

    
    
    training_data = df_features.iloc[train_indices]
    check_data = df_features.iloc[val_indices]
    print('check_data.shape)',check_data.shape)
    max_bound=np.array(training_data.max(axis=0))
    min_bound=np.array(training_data.min(axis=0))
    np.save(f'/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers/{filename}_max_bound.npy',max_bound)
    np.save(f'/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers/{filename}_min_bound.npy',min_bound) 
    
    features = normalise(training_data)
    print('features.shape', features.shape)
    # dp_index=np.where(train_indices==425)[0][0]
    # print(training_data.iloc[dp_index,:])
    # print('normalised oi 425',features[dp_index,:])   
    # sys.exit()
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    # sys.exit()
    cluster_labels = kmeans.labels_
    
    
    ## Count points nearest to each cluster center
    counts = count_nearest_points(check_data, kmeans,max_bound,min_bound)
    
    print(counts, 'validation data split into clusters')
    # Check if each cluster has at least 15 nearest points
    validation = all(len(count) >= 15 for count in counts)
    
    if validation:
        pass
    else:
        save_cluster_files(df_features,n_clusters,retries)
        
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(f'cl{number_clusters}_{filename}.xlsx', engine='xlsxwriter')
    
    # Create dataframes to store mean and std dev of features for each cluster
    mean_df = pd.DataFrame()
    std_df = pd.DataFrame()
    
    # Assuming df_features is the original DataFrame and features is a numpy array derived from df_features
    features_df = pd.DataFrame(features, columns=df_features.columns[:])
    
    # features_df['modified_mask_path'] = df['mask_path'].str[:-5]
    
    features_df['modified_mask_path'] = df[mask_col].iloc[train_indices] 

    for i in range(n_clusters):
        cluster_points_indices = np.where(cluster_labels == i)[0]  # get global indices of all points in cluster i
        cluster_points = features[cluster_points_indices]  # get all points in cluster i
        print('cluster_points.shape',cluster_points.shape)
        cluster_center = kmeans.cluster_centers_[i]  # get the center of cluster i   
        print('cluster_center', cluster_center)
        
        np.save(f'/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers/{filename}_centre_{i}.npy',cluster_center)
        # distances,cov_matrix = mahalanobis_distance(cluster_points, cluster_center)#[distance.euclidean(p, cluster_center) for p in cluster_points] 
        # calculate distances
        distances=[distance.euclidean(p, cluster_center) for p in cluster_points] 
        val_distances=[x[1] for x in counts[i]]
        distances.extend(val_distances)
        
        file_indices=train_indices[cluster_points_indices]
        val_selected_indices=np.array([x[0] for x in counts[i]])
        
        file_indices = np.concatenate((file_indices, val_selected_indices))

        # filepath = df['mask_path'].iloc[cluster_points_indices]
        filepath = df[mask_col][file_indices] 
        # Create a DataFrame
        
        df_sep = pd.DataFrame({
            'Distance': distances,
            'Index': filepath,
            'original index':file_indices
            
        })

        # Sort DataFrame by distance
        df_sep.sort_values('Distance', inplace=True)
        
        
        print(df_sep['Index'].iloc[1],features_df['modified_mask_path'].iloc[1])
        # df_sep.drop(columns=['original index'],inplace=True)
        # Merge the dataframes on the matching columns
        # sys.exit()
        
        result_df = pd.merge(df_sep, df, left_on='Index', right_on='mask_path', how='inner')
        
        # result_df.drop(columns=['modified_mask_path'],inplace=True)
        
        print(df_sep.shape, features_df.shape,result_df.shape)
        assert df_sep.shape[0] == result_df.shape[0],"The shapes of the DataFrames do not match!"
        # corr_df=pd.DataFrame(cov_matrix,index=df_features.columns[:], columns=df_features.columns[:])
        # plot_heatmap(i, corr_df)
        # plot_bubble_correlation(i,corr_df)
        # plot_pairgrid(i,corr_df)
        # Calculate mean and std dev for each feature in this cluster
        mean_series = result_df.mean(numeric_only=True)
        std_series = result_df.std(numeric_only=True)
        
        
        # Add these as new columns to the mean and std dev dataframes
        mean_df[f'Cluster_{i}'] = mean_series
        std_df[f'Cluster_{i}'] = std_series
       

        
        # Write each DataFrame to a specific sheet
        result_df.to_excel(writer, sheet_name=f'Cluster_{i}', index=False)
        # corr_df.to_excel(writer, sheet_name=f'ClusterCorr_{i}', index=True)
        
    
    mean_df.drop(['original index'],inplace=True)
    std_df.drop(['original index'],inplace=True)

    # Write mean and std dev dataframes to separate sheets
    mean_df.to_excel(writer, sheet_name='Mean_Features', index=True)
    std_df.to_excel(writer, sheet_name='Std_Dev_Features', index=True)
    CV_df = std_df/mean_df
    CV_df.to_excel(writer,sheet_name='Coefficient of variation',index=True)
    
    # Save the excel
    writer.save()
    
    return 'Files saved successfully'

# sil_plots(df_features)
    
save_cluster_files(df_features)


