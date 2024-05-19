import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from Input.dataset import test_indices,train_indices, val_indices
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# Load the Excel file
# file_path = '/scratch/a.bip5/BraTS/TrainExpertClusterExp_7763883.xlsx' 

file_path2= '/scratch/a.bip5/BraTS/Itemised Dice/Expert_eval_full_set/output/MFP_TrainExpertClusterExp_7763883.xlsx'# Update with your file path
# excel_data = pd.read_excel(file_path, sheet_name='Everything')
excel_data2 = pd.read_excel(file_path2, sheet_name='Everything')

# excel_data=excel_data.dropna()
# excel_data2=excel_data2.replace(0,pd.NA).dropna()

columns_to_average = [
    'sagittal_profile_tc','frontal_profile_tc','axial_profile_tc','sagittal_profile_wt','frontal_profile_wt','axial_profile_wt','sagittal_profile_et','frontal_profile_et','axial_profile_et'
        ]
        
    # we want to drop all the samples where one or more tumor category is missing to remove noise from later analysis     
for column in columns_to_average:
    excel_data2 = excel_data2[excel_data2[column]!=0]
 
# # Preprocessing: Imputing NaN values with the median
# for col in ['avg_dice_profile', 'avg_delta_area', 'avg_regularity']:
    # # median_value = excel_data[col].median()
    # median_value2 = excel_data2[col].median()
    # # excel_data[col].fillna(median_value, inplace=True)
    # excel_data2 = excel_data2.replace(0, np.nan).fillna(median_value2)
    




#making sure data is sorted to use iloc indices 
excel_data2 = excel_data2.sort_values(by='Unnamed: 0_x') 

print(excel_data2.iloc[0:2,:])

    # excel_data2[col].fillna(median_value2, inplace=True)
# train_indices=np.concatenate((train_indices, val_indices))

# Selecting the relevant columns for input features and target variable
input_features = ['average dprofile']# 'avg_dice_profile','avg_delta_area','dmin', 'avg_regularity', 'Expert', 'd0', 'd1', 'd2', 'd3','average_profile',,'average dprofile',,'average dprofile','average_profile'
target = 'average'
# X = excel_data[input_features]

# y = excel_data[target]*100




# print(X.sample(n=5,random_state=randomstate))
# print(X2.sample(n=5,random_state=randomstate))
# # print(y.sample(n=5,random_state=randomstate))
# print(y2.sample(n=5,random_state=randomstate))



# Splitting the dataset into training, validation, and test sets
# Assuming train_indices, val_indices, and test_indices are defined
train_indices=np.concatenate((train_indices, val_indices))

filt_train = np.isin(train_indices, excel_data2['Unnamed: 0_x'])

train_indices_filt = train_indices[filt_train] #getting rid indices that have been dropped


# X_train = X.iloc[train_indices]
# y_train = y.iloc[train_indices]

# X_val = X.iloc[val_indices]
# y_val = y.iloc[val_indices]

# X_test = X.iloc[np.sort(test_indices)]
# y_test = y.iloc[np.sort(test_indices)]

print('train indices length, max and min', len(train_indices_filt), max(train_indices_filt), min(train_indices_filt))

# print('X2 shape', X2.shape) 
# print('y2 shape', y2.shape) 


X_train2 = excel_data2[excel_data2['Unnamed: 0_x'].isin(train_indices_filt)][input_features]

y_train2= excel_data2[excel_data2['Unnamed: 0_x'].isin(train_indices_filt)][target]*100

print(X_train2.head())

randomstate=np.random.randint(213)
# X_train2 = X2.iloc[train_indices_filt]
# y_train2 = y2.iloc[train_indices_filt]

# equal=[0,0]
# notequal=[0,0]
# print(X_train.shape,X_train2.shape)

# for index,(x1,yi1,x2,yi2) in enumerate(zip(X_train.to_numpy(),y_train.to_numpy(),X_train2.to_numpy(), y_train2.to_numpy())):
    
    # if (x1-x2)<0.00000001:
        
        # equal[0]+=1
    # else: 
        # print('not equal')
        # print(x1,x2)
        # notequal[0]+=1
    # if (yi1-yi2)<0.000001:
        # equal[1]+=1
    # else:        
        # notequal[1]+=1

# print('equal, notequal',equal, notequal)

filt_test = np.isin(test_indices, excel_data2['Unnamed: 0_x'])

test_indices_filt=test_indices[filt_test] #getting rid indices that have been dropped

print('test indices length, max and min', len(test_indices_filt), max(test_indices_filt), min(test_indices_filt))

X_test2 =excel_data2[excel_data2['Unnamed: 0_x'].isin(test_indices_filt)][input_features]
y_test2 = excel_data2[excel_data2['Unnamed: 0_x'].isin(test_indices_filt)][target]*100

print(X_test2.head())
# X_t = X.iloc[np.sort(test_indices)]
# y_t = y.iloc[np.sort(test_indices)]

# X_train,x_rem,y_train,y_rem=train_test_split(X_t,y_t,test_size=0.4,random_state=0)
# X_val,X_test,y_val,y_test=train_test_split(x_rem,y_rem,test_size=0.5,random_state=21)

# Add a constant to the model for the intercept if it's not included in X_train
# X_train_with_const = sm.add_constant(X_train)
# model = sm.OLS(y_train, X_train_with_const)
# results = model.fit()
# print(results.summary())

# # Add a constant to X_test if your model includes an intercept
# X_test_with_const = sm.add_constant(X_test)

# # Use the model to make predictions on the test set
# y_pred = results.predict(X_test_with_const)

# ss_res = ((y_test - y_pred) ** 2).sum()
# ss_tot = ((y_test - y_test.mean()) ** 2).sum()
# r_squared = 1 - (ss_res / ss_tot)
# # Calculate MSE
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# # Linear Regression Model
# linear_reg = LinearRegression()
# linear_reg.fit(X_train, y_train)
# y_pred_lr = linear_reg.predict(X_test)
# mse_lr = mean_squared_error(y_test, y_pred_lr)
# r2_lr = r2_score(y_test, y_pred_lr)


linear_reg2 = LinearRegression()
linear_reg2.fit(X_train2, y_train2)
y_pred_lr2 = linear_reg2.predict(X_test2)
mse_lr2 = mean_squared_error(y_test2, y_pred_lr2)
r2_lr2 = r2_score(y_test2, y_pred_lr2)
# print("Linear Regression - RMSE:", np.sqrt(mse_lr), "R2:", r2_lr)
print("Linear Regression - RMSE2:", np.sqrt(mse_lr2), "R2:", r2_lr2)
# Specify the filename
# filename = "regression_summary.txt"

# # Open the file in write mode ('w') and write the metrics
# with open(filename, 'w') as file:
    # file.write(f"R-squared: {r_squared}\n")
    # file.write(f"RMSE: {rmse}\n")
    # file.write(results.summary().as_text())


# print(f"Results saved to {filename}")
# MLP Model
# Converting data to PyTorch tensors
# X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
# X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# # Creating TensorDatasets and DataLoaders
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor.view(-1, 1))
# val_dataset = TensorDataset(X_val_tensor, y_val_tensor.view(-1, 1))
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor.view(-1, 1))
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Defining the MLP model
# class MLP(nn.Module):
    # def __init__(self):
        # super(MLP, self).__init__()
        # self.layers = nn.Sequential(
            # nn.Linear(4, 64),
            # nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            # nn.Linear(64, 1)
        # )

    # def forward(self, x):
        # return self.layers(x)

# mlp_model = MLP()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# # Training the MLP model
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    # for epoch in range(num_epochs):
        # model.train()
        # for inputs, targets in train_loader:
            # optimizer.zero_grad()
            # outputs = model(inputs)
            # loss = criterion(outputs, targets)
            # loss.backward()
            # optimizer.step()

        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
            # for inputs, targets in val_loader:
                # outputs = model(inputs)
                # loss = criterion(outputs, targets)
                # val_loss += loss.item()
        # val_loss /= len(val_loader)
        # if (epoch+1) % 10 == 0:
            # print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

# train_model(mlp_model, train_loader, val_loader, criterion, optimizer, num_epochs=100)

# # Evaluating the MLP model
# def evaluate_model(model, test_loader, criterion):
    # model.eval()
    # test_loss = 0.0
    # predictions = []
    # actuals = []
    # with torch.no_grad():
        # for inputs, targets in test_loader:
            # outputs = model(inputs)
            # loss = criterion(outputs, targets)
            # test_loss += loss.item()
            # predictions.extend(outputs.view(-1).tolist())
            # actuals.extend(targets.view(-1).tolist())
    # test_loss /= len(test_loader)
    # return test_loss, predictions, actuals

# test_loss_mlp, predictions_mlp, actuals_mlp = evaluate_model(mlp_model, test_loader, criterion)
# mse_mlp = mean_squared_error(actuals_mlp, predictions_mlp)
# r2_mlp = r2_score(actuals_mlp, predictions_mlp)

# Print performance metrics

# print("MLP - RMSE:", np.sqrt(mse_mlp), "R2:", r2_mlp)
