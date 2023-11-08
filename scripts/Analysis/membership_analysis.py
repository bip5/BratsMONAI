import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cnn_clusters='/scratch/a.bip5/BraTS 2021/selected_files_seed1695670857_cnn.xlsx'
manual_clusters='/scratch/a.bip5/BraTS 2021/selected_files_seed1694641315MM.xlsx'

cnn_xls=pd.ExcelFile(cnn_clusters)
man_xls=pd.ExcelFile(manual_clusters)

sheet_names = cnn_xls.sheet_names
sheet_names=[x for x in sheet_names if 'Cluster' in x]


# Function to read and process sheets from an excel file
def process_sheets(xls, sheet_names):
    dfs = []
    for sheet in sheet_names:
        # Read sheet
        df = pd.read_excel(xls, sheet_name=sheet)
        # Assign cluster number
        cluster_number = int(sheet.split('_')[-1])
        df['cluster'] = cluster_number
        # Keep only path and cluster columns
        df = df[['Index', 'cluster']]
        # Set index to 'path'
        
        dfs.append(df)
    return pd.concat(dfs)

# Process excel files
cnn_df = process_sheets(cnn_clusters, sheet_names)

man_df = process_sheets(manual_clusters, sheet_names)
man_df.reset_index(inplace=True)


man_df['Index'] = man_df['Index'].str.replace(r'^\./', '/scratch/a.bip5/BraTS 2021/', regex=True)
man_df.set_index('Index', inplace=True)
cnn_df.set_index('Index', inplace=True)
   
# Merge dataframes on 'path'
merged_df = cnn_df.merge(man_df, left_index=True, right_index=True, suffixes=('_cnn', '_manual'))


merged_df.to_csv('merged_df.csv')

plt.figure(figsize=(12, 6))
for idx, row in merged_df.iterrows():
    plt.plot(['CNN', 'Manual'], [row['cluster_cnn'], row['cluster_manual']], marker='o', linestyle='-')

plt.ylabel('Cluster Number')
plt.title('Cluster Membership: CNN vs Manual')
plt.savefig('membership comparison.jpg')
plt.close()