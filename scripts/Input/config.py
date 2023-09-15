root_dir = "/scratch/a.bip5/BraTS 2021/BraTS21_data"
model_name = 'SegResNet' # pick from UNet SegResNet
total_epochs = 500
val_interval = 1
VAL_AMP = True
load_save = 1
seed = 0
load_path = '/scratch/a.bip5/BraTS 2021/BraTS21_data/SegResNetCV1ms1250rs0A'
fold_num = 1
max_samples = 1250
method = 'A'#type of ensemble method
upsample = 'DECONV' # upsample method in SegResNet
batch_size = 4
T_max = 100 #how often to reset cycling
workers = 8
lr = 0.0002
fs_ensemble = 0 # fewshot ensemble
exp_ensemble = 1 #'expert' ensemble
CV_flag = 1 if fs_ensemble+exp_ensemble==0 else 0
DDP = False
train_partial = False
unfreeze = 4 #set depending on how many items to unfreeze from layer list
freeze_train=True # start freezing layers in training if freeze criteria met
isolate_layer=True #whether to isolate one layer at a time while freeze training
lr_cycling=False
cluster_files ='/scratch/a.bip5/BraTS 2021/selected_files_seed1694621970MM.xlsx'
################EVAL SPECIFIC VARIABLES#################
eval_path = '/scratch/a.bip5/BraTS 2021/BraTS21_data/'+'2023-09-14T1694703670.5156782Cluster_0SegResNet' 

eval_mode = 'cluster' #choose between ensemble, cluster and so on.

# unused_collection=True if fs_ensemble==0 else False