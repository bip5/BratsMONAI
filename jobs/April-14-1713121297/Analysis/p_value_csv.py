import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,variation,ttest_rel
from statsmodels.stats.anova import AnovaRM
import os
import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')

from Input.config import plots_dir

filename1 = 'ens_valexpert_time_samp_7785266'
final_df1 = pd.read_csv(f'/scratch/a.bip5/BraTS/{filename1}.csv')
final_df1['model'] = filename1

filename2 = 'valexpert_time_samp_7785267'
final_df2 = pd.read_csv(f'/scratch/a.bip5/BraTS/{filename2}.csv')
final_df2['model'] = filename2

p_value_dice = ttest_rel(final_df1['d average'],final_df2['d average'] )

p_value_volume = ttest_rel(final_df1['predicted volume delta'].abs(),final_df2['predicted volume delta'].abs() )

print('p_value_dice validation expert ensemble vs expert', p_value_dice)
print('p_value_volume absolute delta, val ensemble vs expert',p_value_volume)


filename3 = 'Ensembleexpert_time_samp_7785577'
final_df3 = pd.read_csv(f'/scratch/a.bip5/BraTS/{filename3}.csv')
final_df3['model'] = filename3

filename4 = 'stdexpert_time_samp_7782487'
final_df4 = pd.read_csv(f'/scratch/a.bip5/BraTS/{filename4}.csv')
final_df4['model'] = filename4

p_value_dice_std =ttest_rel(final_df3['d average'],final_df4['d average'] )

p_value_volume_std =ttest_rel(final_df3['predicted volume delta'].abs(),final_df4['predicted volume delta'].abs() )

print('p_value_dice std expert ensemble vs expert', p_value_dice_std)
print('p_value_volume absolute delta, std ensemble vs expert',p_value_volume_std)

base_model='basemodel_time_samp_7781810'
final_df5=pd.read_csv(f'/scratch/a.bip5/BraTS/{base_model}.csv')
final_df5['model'] = base_model

df_long = pd.concat([final_df1,final_df2,final_df3,final_df4,final_df5], ignore_index=True)

aovrm = AnovaRM(df_long, 'd average', 'level_0_old', within=['model'])
res = aovrm.fit()

print(res.summary())

aovrm = AnovaRM(df_long, 'predicted volume delta', 'level_0_old', within=['model'])
res = aovrm.fit()

print(res.summary())