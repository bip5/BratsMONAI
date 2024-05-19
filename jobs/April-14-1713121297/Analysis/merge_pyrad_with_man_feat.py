import pandas as pd

# File paths
file_path_1 = '/scratch/a.bip5/BraTS/Dataset_features_fixed1697571928.csv'
file_path_2 = '/scratch/a.bip5/BraTS/prd_feat_2023-12-08_05.csv'

# Read the datasets
df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)

# Assuming 'subject_id' column exists in df2 and is the index after loading
df2['subject_id']=df2['Unnamed: 0']

df2.drop(['Unnamed: 0'], axis=1, inplace=True)
# Drop the diagnostic columns from the second dataframe
# Assuming these are columns with 'diagnostic' in their name
diagnostic_columns = [col for col in df2.columns if 'diagnostic' in col]
df2.drop(diagnostic_columns, axis=1, inplace=True)

# Drop specified columns from the first dataframe
columns_to_drop_from_first = [
    'axial0_contrast','axial0_dissimilarity','axial0_homogeneity','axial0_energy','axial0_correlation','coronal0_contrast','coronal0_dissimilarity','coronal0_homogeneity','coronal0_energy','coronal0_correlation','sagittal0_contrast','sagittal0_dissimilarity','sagittal0_homogeneity','sagittal0_energy','sagittal0_correlation','axial1_contrast','axial1_dissimilarity','axial1_homogeneity','axial1_energy','axial1_correlation','coronal1_contrast','coronal1_dissimilarity','coronal1_homogeneity','coronal1_energy','coronal1_correlation','sagittal1_contrast','sagittal1_dissimilarity','sagittal1_homogeneity','sagittal1_energy','sagittal1_correlation','axial2_contrast','axial2_dissimilarity','axial2_homogeneity','axial2_energy','axial2_correlation','coronal2_contrast','coronal2_dissimilarity','coronal2_homogeneity','coronal2_energy','coronal2_correlation','sagittal2_contrast','sagittal2_dissimilarity','sagittal2_homogeneity','sagittal2_energy','sagittal2_correlation','axial3_contrast','axial3_dissimilarity','axial3_homogeneity','axial3_energy','axial3_correlation','coronal3_contrast','coronal3_dissimilarity','coronal3_homogeneity','coronal3_energy','coronal3_correlation','sagittal3_contrast','sagittal3_dissimilarity','sagittal3_homogeneity','sagittal3_energy','sagittal3_correlation','Unnamed: 0'
]
df1.drop(columns_to_drop_from_first, axis=1, inplace=True)

# Create an additional column by averaging the values in the non-nan rows from the specified columns
columns_to_average = [
    'sagittal_profile_tc','frontal_profile_tc','axial_profile_tc','sagittal_profile_wt','frontal_profile_wt','axial_profile_wt','sagittal_profile_et','frontal_profile_et','axial_profile_et'
]
df1['average_profile'] = df1[columns_to_average].mean(axis=1, skipna=True)

# Extract subject IDs from the 'mask_path' column in the first dataframe
# Assuming the subject ID is the last part of the path in 'mask_path'
df1['subject_id'] = df1['mask_path'].apply(lambda x: x.split('/')[-1][:-11])

# Merge the two dataframes on 'subject_id'
result = pd.merge(df1, df2, left_on='subject_id',  right_on="subject_id")

columns=result.columns.tolist()
columns.remove('subject_id')
columns=['subject_id']+columns
result=result[columns]

# Drop the 'mask_path' column from the result
# result.drop('mask_path', axis=1, inplace=True)

# Assuming you want to save the result to a new CSV file
result.to_csv('/scratch/a.bip5/BraTS/merged_features.csv', index=True)
