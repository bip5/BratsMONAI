import pandas as pd
import numpy as np
from scipy.stats import pearsonr

file=pd.read_csv('/scratch/a.bip5/BraTS/manual_features_prf_2023-11-27_16.csv')

''' create separate Dataframes for each cluster '''
cluster_0=file[file['Cluster']='Cluster_0']
cluster_1=file[file['Cluster']='Cluster_1']
cluster_2=file[file['Cluster']='Cluster_2']
cluster_3=file[file['Cluster']='Cluster_3']

''' we want to create a table where we can see all possible correlations.
One important question we were trying to answer was: 
Is there anything specific about low dice performances? The data at a glance says no, but we need to support it with pvalue, for this reason we need to calculate p-value for all columns one at a time against performance.

The format
feature ef size: correlation, p-value

 '''
 c_p=dict()
for column in cluster_0.columns[19:]:
    c,p=pearsonr(cluster_0[column],cluster_0['SegResNetCV1_j7688198ep128'])

    c_p[column]=[c,p]

correlation_pvalue=pd.DataFrame(c_p,columns=['Correlation', 'p-value'])
