import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,variation
import os
import sys
# Read in 

now=datetime.datetime.now().strftime('%Y-%m-%d_%H')
filename='/scratch/a.bip5/BraTS/cl4_B23al_2023-11-21_16-20-10.xlsx'

opened_file = pd.ExcelFile(filename)
    # Get all sheet names
sheet_names = opened_file.sheet_names
sheet_names=[x for x in sheet_names if 'Cluster_' in x]

for sheet in sheet_names:
    cluster=opened_file.parse(sheet)
    cluster.drop(cluster.columns[0:3],axis=1,inplace=True)
    cluster_center=np.array(cluster.mean(axis=0))
    np.save(f'/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers/{sheet}.npy',cluster_center)