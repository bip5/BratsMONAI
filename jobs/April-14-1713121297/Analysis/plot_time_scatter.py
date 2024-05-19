import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,variation,ttest_rel
import os
import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')
import pingouin as pg
from statsmodels.stats.anova import AnovaRM
from Evaluation.eval_functions import plot_expert_performance

from Input.config import plots_dir

filenames=['stdexpert_time_samp_7782487','Ensembleexpert_time_samp_7785577','valexpert_time_samp_7785267','ens_valexpert_time_samp_7785266','basemodel_time_samp_7781810']
dfs=[]
for filename in filenames:
    print(filename)
    final_df=pd.read_csv(f'/scratch/a.bip5/BraTS/{filename}.csv')
    final_df['growth_marker']=np.where(final_df['GT delta']>0,'red','green')
    final_df['filename']=filename
    dfs.append(final_df)
    # job_id = filename.split('_')[2].split('.')[0]
    # print(job_id)
    # print(final_df.head())
    save_path= os.path.join(plots_dir,f'pvd_{filename}.eps')
    plt.figure()
    plt.scatter(final_df['GT delta'],final_df['predicted volume delta'],c=final_df['growth_marker'],alpha=0.5)
    plt.xlabel('Volume change actual(mm$^3$)')
    plt.ylabel('Prediction Error(mm$^3$)')
    plt.ylim(-100000,100000)
    plt.xlim(-200000,170000)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)

    plt.close()
    
    
    growth_df = final_df[final_df['growth_marker'] == 'red']
    shrinkage_df = final_df[final_df['growth_marker'] == 'green']

    plt.figure()

    # Plotting each category separately to display legend
    plt.scatter(growth_df['average_GTGT'], growth_df['d average'], color='red', alpha=0.5, label='Growth')
    plt.scatter(shrinkage_df['average_GTGT'], shrinkage_df['d average'], color='green', alpha=0.5, label='Shrinkage')

    plt.xlabel('Dice score between the two Ground Truths')
    plt.ylabel('Dice score for delta volume prediction')
    plt.ylim(0.4, 1)
    plt.xlim(0, 1)
    plt.grid(True)
    plt.tight_layout()

    # Add the legend.
    plt.legend()

    # Save the figure.
    save_path = os.path.join(plots_dir, f'deltaDice_{filename}.eps')
    plt.savefig(save_path)


    # p_value_oldnew =ttest_rel(final_df['average_oldSnewT'],final_df['average_GTGT'] )

    # p_value_newold=ttest_rel(final_df['average_newSoldT'],final_df['average_GTGT'] )

    # print(f'p_value_oldSnewT: {p_value_oldnew}, p_value_newSoldT : {p_value_newold}')

    # p_value_oldnew_pr =pearsonr(final_df['average_oldSnewT'],final_df['average_GTGT'] )

    # p_value_newold_pr=pearsonr(final_df['average_newSoldT'],final_df['average_GTGT'] )

    # print(f'p_value_oldSnewT_pr: {p_value_oldnew_pr}, p_value_newSoldT_pr : {p_value_newold_pr}')

    aov=pg.welch_anova(dv='d average', between='marker_color',data=final_df)
    print( aov)
    
    aov2=pg.welch_anova(dv='predicted volume delta', between='marker_color',data=final_df)
    print('volume change anova\n', aov2)
    
filenames2=['IndScoresm2024-01-17_18-32-16_7749907_stdExpert','IndScoresm2024-01-19_20-34-43_7751753_scratchExp','IndScoresm2024-01-22_19-23-02_7754201_valExpert','stdDistEns','scratch_distENs', 'valDistEns']

for filename in filenames2:
    print(filename)
    ind_score_df=pd.read_excel(f'/scratch/a.bip5/BraTS/{filename}.xlsx',sheet_name='Everything')
    save_path=f'/scratch/a.bip5/BraTS/plots/{filename}.eps'
    plot_expert_performance(ind_score_df,save_path,plot_together=True)
    
# df_long=pd.concat(dfs, ignore_index=True)
# print(df_long.shape, 'dflong shape')
# rm=AnovaRM(df_long,'predicted volume delta','level_0_old',within=['filename'])
# method_comparison_volume=rm.fit()
# print('volume change aov_method_comparison\n', method_comparison_volume.summary())

# rm2=AnovaRM(df_long,'d average','level_0_old',within=['filename'])
# method_comparison_dice=rm2.fit()
# print('delta dice aov_method_comparison\n', method_comparison_dice.summary())

# rm2=AnovaRM(df_long,'d tc','level_0_old',within=['filename'])
# method_comparison_dice=rm2.fit()
# print('delta dice aov_method_comparison\n', method_comparison_dice.summary())

# rm2=AnovaRM(df_long,'d wt','level_0_old',within=['filename'])
# method_comparison_dice=rm2.fit()
# print('delta dice aov_method_comparison\n', method_comparison_dice.summary())

# rm2=AnovaRM(df_long,'d et','level_0_old',within=['filename'])
# method_comparison_dice=rm2.fit()
# print('delta dice aov_method_comparison\n', method_comparison_dice.summary())


# for i in range(5):
    # for j in range(5):
     
        # p_value_vol = ttest_rel(dfs[i]['predicted volume delta'],dfs[j]['predicted volume delta'] )
        # print(f'p value vol for {filenames[i]} against {filenames[j]}:, {p_value_vol}')
        
# for i in range(5):
    # for j in range(5):
        # p_value_dice = ttest_rel(dfs[i]['d average'],dfs[j]['d average'] )
        # print(f'p value dice for {filenames[i]} against {filenames[j]}:, {p_value_dice}')