import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')
from Input.dataset import test_indices,train_indices, val_indices
import statsmodels.api as sm
from time import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load the Excel file to read in the dataset to predict performance
file_path = '/scratch/a.bip5/BraTS/Itemised Dice/Expert_eval_full_set/output/MFP_TrainExpertClusterExp_7763883.xlsx'#'/scratch/a.bip5/BraTS/TrainExpertClusterExp_7763883.xlsx'  # Update with your file path
df = pd.read_excel(file_path, sheet_name='Everything')
print(df.columns)
# Drop unwanted columns
columns_to_exclude = ['tc', 'wt', 'et', 'Subject ID']
columns_to_exclude += [col for col in df.columns if 'Unnamed' in col]
df_filtered = df.drop(columns=columns_to_exclude)

# Preprocessing: Imputing NaN values with the median
for col in ['avg_dice_profile', 'avg_delta_area', 'avg_regularity']:
    # median_value = excel_data[col].median()
    median_value2 = df_filtered[col].median()
    # excel_data[col].fillna(median_value, inplace=True)
    df_filtered = df_filtered.replace(0, np.nan).fillna(median_value2)

scaler=MinMaxScaler()


# Step 1: Prepare the data
X = df_filtered.drop(columns=['average'])  # Features matrix
X=pd.DataFrame(scaler.fit_transform(X),columns=X.columns, index=X.index)
y = df_filtered['average']  # Target variable

# Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_indices=np.concatenate((train_indices, val_indices))
X_train = X.iloc[train_indices]
y_train = y.iloc[train_indices]



X_test = X.iloc[np.sort(test_indices)]
y_test = y.iloc[np.sort(test_indices)]

# Step 2: Feature selection with LassoCV
lasso = LassoCV(cv=5, random_state=42, alphas=np.logspace(-6, 6, 13))
lasso.fit(X_train, y_train)

# Print selected features
print("Features selected by LassoCV:", list(X.columns[lasso.coef_ != 0]))

# Step 3: Train a Linear Regression model using only selected features


selected_features = X.columns[lasso.coef_ != 0]
X_train=X_train[selected_features]

X_test=X_test[selected_features]

X_train_with_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_with_const)
results = model.fit()
print(results.summary())

# Add a constant to X_test if your model includes an intercept
X_test_with_const = sm.add_constant(X_test)

# Use the model to make predictions on the test set
y_pred = results.predict(X_test_with_const)

ss_res = ((y_test - y_pred) ** 2).sum()
ss_tot = ((y_test - y_test.mean()) ** 2).sum()
r_squared = 1 - (ss_res / ss_tot)
# Calculate MSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Linear Regression Model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_lr = linear_reg.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Specify the filename
filename = f"regression_summary{str(time())[-3:]}.txt"

# Open the file in write mode ('w') and write the metrics
with open(filename, 'w') as file:
    file.write(f"R-squared: {r_squared}\n")
    file.write(f"RMSE: {rmse}\n")
    file.write(results.summary().as_text())


print(f"Results saved to {filename}")
print("Linear Regression - RMSE:", np.sqrt(mse_lr), "R2:", r2_lr)