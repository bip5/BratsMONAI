import pandas as pd
from scipy import stats

def calculate_ttest_rel(file_path1, file_path2):
    # Constant root directory
    root = '/scratch/a.bip5/BraTS/'
    
    # Concatenate root with the provided file paths
    full_file_path1 = f'{root}{file_path1}'
    full_file_path2 = f'{root}{file_path2}'
    
    # Read in the excel files
    e1 = pd.read_excel(full_file_path1, sheet_name='Everything')
    e2 = pd.read_excel(full_file_path2, sheet_name='Everything')
    
    # Perform the paired t-test on the 'average' column
    t_statistic, p_value = stats.ttest_rel(e1['average'], e2['average'])
    
    return t_statistic, p_value

filepath1='IndEnsemblem2024-01-17_18-32-16_7769525_plainEns.xlsx'
filepath2='Itemised Dice/ExpertModels/stdDistEns.xlsx'
t_statistic, p_value = calculate_ttest_rel(filepath1, filepath2)
print(f'T-statistic: {t_statistic}, P-value: {p_value}')
