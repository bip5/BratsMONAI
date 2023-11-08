import pandas as pd

# Read in the CSV files into dataframes
df1 = pd.read_csv('IndScoresm2023-10-19_21-38-43.csv', index_col=0)
df2 = pd.read_csv('Dataset_features_fixed1697571928.csv')

# Convert the index of df1 to a column
df1 = df1.reset_index().rename(columns={'index': 'subject_id'})

# Create a temporary key column containing the mask path in df1
df1['key'] = df1['subject_id'].apply(lambda x: df2['mask_path'].str.contains(x).idxmax() if df2['mask_path'].str.contains(x).any() else None)
merged_df = pd.merge(df1, df2, left_on='key', right_index=True, how='inner').drop(columns='key')


# Save the new dataframe if needed
merged_df.to_csv('perf_with_manual_features.csv', index=False)
