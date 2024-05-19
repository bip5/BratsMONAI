import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,variation
import os
import sys
'''Script to combine the file with manual features called merged_features with output.'''

now=datetime.datetime.now().strftime('%Y-%m-%d_%H')
source_path='/scratch/a.bip5/BraTS/Itemised Dice/Expert_eval_full_set/'#'/scratch/a.bip5/BraTS/Itemised Dice/ExpertModels/'

#processing in a loop for a given folder so we can process more than one file in a single run
for filename in os.listdir(source_path):
    if not 'xls' in filename:
        continue
    individual_scores_path = source_path + f'{filename}'    
    output_dir = os.path.join(os.path.join('/',*source_path.split('/')[:-1]),'output/')
    print('output_dir', output_dir)
    os.makedirs(output_dir,exist_ok=True)
    savename = 'MFP_' + individual_scores_path.split('/')[-1]
    writer = pd.ExcelWriter(f'{output_dir}{savename}',engine='xlsxwriter')
    plot_dir = source_path.split('/')[-1]    
    new_dir = os.path.join('/scratch/a.bip5/BraTS/plots', f'{plot_dir}')
    os.makedirs(new_dir, exist_ok=True)
    file = pd.ExcelFile(individual_scores_path)

    sheets=file.sheet_names
    sheets=['Everything']+[x for x in sheets if 'Expert' in x]
    cluster_dfs=dict()
    for sheet in sheets:

        df1=pd.read_excel(individual_scores_path,sheet_name=sheet)
        df2 = pd.read_csv('/scratch/a.bip5/BraTS/merged_features.csv')


        # Create a temporary key column containing the mask path in df1
        # df1['key'] = df1['Subject ID'].apply(lambda x: df2['mask_path'].str.contains(x).idxmax() if df2['mask_path'].str.contains(x).any() else None)
        # merged_df = pd.merge(df1, df2, left_on='key', right_index=True, how='inner').drop(columns=['key','Unnamed: 0'])
        
       
        df2['key']=df2['subject_id'].map(lambda x: x[10:]) #making key same as df1
        print(df1.columns)
        merged_df = pd.merge(df1, df2, left_on='Subject ID', right_on='key', how='inner').drop(columns=['subject_id','key','Unnamed: 0','mask_path'],errors='ignore')
        
        # print(merged_df['Subject ID'][:10],merged_df['subject_id'][:10], ' SHAPPEEEEE') ##here to check before dropping subject_id column
        # print('SYS EXITING REMOVE LINE IF NOT NEEDED')
        # sys.exit()
        merged_df.fillna(0,inplace=True)
        columns_to_average = [
        'sagittal_profile_tc','frontal_profile_tc','axial_profile_tc','sagittal_profile_wt','frontal_profile_wt','axial_profile_wt','sagittal_profile_et','frontal_profile_et','axial_profile_et'
            ]
        #we want to drop all the samples where one or more tumor category is missing to remove noise from later analysis     
        for column in columns_to_average:
            merged_df=merged_df[merged_df[column]!=0]
            
        merged_df['average dprofile']=merged_df[columns_to_average].mean(axis=1)
        # for x in merged_df.columns:
            # print(x)
        # sys.exit()
        
        orig_c_p=dict()
        orig_c_p['Correlation']=[]
        orig_c_p['p_value']=[]
        orig_c_p['Average Value']=[]
        orig_c_p['CoV']=[]
        orig_c_p['max']=[]
        orig_c_p['min']=[]
        
        # Below we're gathering information for the summary sheet in the output file
        for column in merged_df.columns[6:]:
            cm,pm=pearsonr(merged_df[column],merged_df['average'])#'SegResNetCV1_j7688198ep128'
            a=merged_df[column].mean()
            mx=merged_df[column].max()
            mn=merged_df[column].min()
            cov=variation(merged_df[column])
            orig_c_p['Correlation'].append(cm)
            orig_c_p['p_value'].append(pm)
            orig_c_p['Average Value'].append(a)
            orig_c_p['max'].append(mx)
            orig_c_p['min'].append(mn)
            orig_c_p['CoV'].append(cov)
        
        plot_dfs = {} 
        for i,cluster in enumerate(merged_df['Expert'].unique()):
            cluster_df=merged_df[merged_df['Expert']==cluster]
            cluster_df.fillna(0,inplace=True)
            c_p=dict()
            c_p['Correlation']=[]
            c_p['p_value']=[]
            c_p['Average Value']=[]
            c_p['CoV']=[]
            c_p['max']=[]
            c_p['min']=[]
            columns=[]
            # Filter columns based on cluster name or specific column
            # relevant_columns = [col for col in cluster_df.columns[3:8] if cluster in col or col == 'Base Average Dice']
            # plot_df = cluster_df[relevant_columns]
            # remaining_column=plot_df.columns.tolist()
            # remaining_column.remove('Base Average Dice')
            
            # plot_df['Delta_Performance'] = plot_df[remaining_column[0]]-plot_df['Base Average Dice'] 
            # plt.scatter(plot_df['Base Average Dice'], plot_df['Delta_Performance'], label=f'{cluster}') 
            
            for column in cluster_df.columns[6:]:
                              
                c,p=pearsonr(cluster_df[column],cluster_df['average'])
                a=cluster_df[column].mean()
                mx=cluster_df[column].max()
                mn=cluster_df[column].min()
                cov=variation(cluster_df[column])
                c_p['Correlation'].append(c)
                c_p['p_value'].append(p)
                c_p['Average Value'].append(a)
                c_p['max'].append(mx)
                c_p['min'].append(mn)
                c_p['CoV'].append(cov)
                
                columns.append(column)
            cluster_dfs['orig'+sheet+str(cluster)]=cluster_df
            cluster_dfs[sheet+str(cluster)]=pd.DataFrame(c_p,index=columns).sort_values(by=['Correlation'],axis=0,ascending=False)
            cluster_dfs[sheet +' summary']=pd.DataFrame(orig_c_p,index=columns).sort_values(by=['Correlation'],axis=0,ascending=False)
         # Finalizing plot for the sheet
        # plt.title(f'Dice after training {str(filename)[:-5]}')
        # plt.xlabel('Base Model Performance')
        # plt.ylabel('Delta Performance')
        # plt.ylim(-0.5,0.13)
        # plt.grid()
        # plt.axhline(y=0, color='pink', linestyle='-', alpha=0.5)
        # plt.legend()
        # plt.savefig(f'{new_dir}/{sheet}_{filename[:-5]}.png')
        # plt.clf()
        
        merged_df.to_excel(writer,sheet_name=sheet,index=False)
        print(cluster_dfs.keys())
        for key in cluster_dfs.keys():
            cluster_dfs[key].to_excel(writer,sheet_name=key,index=True)
      
    writer.close() #same as writer.save()

