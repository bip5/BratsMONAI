batch_size = 4
cluster_files ='/scratch/a.bip5/BraTS/selected_files_seed1695670857_cnn.xlsx'
dropout=0
exp_train_count=5
exp_val_count=20
freeze_patience=5
fold_num = 1
#set depending on how many items to unfreeze from layer list
load_save = 1
load_path = '/scratch/a.bip5/BraTS/weights/m2023-10-10_19-21-14/SegResNetCV1ms1250rs0Aep273'
lr = 0.0002
root_dir = "/scratch/a.bip5/BraTS/BraTS_23_training"
weights_dir="/scratch/a.bip5/BraTS/weights"
max_samples = 1251
method = 'A'#type of ensemble method
model_name = 'SegResNet' # pick from UNet SegResNet
seed = 0
T_max = 300 #how often to reset cycling
total_epochs = 300
unfreeze = 22 
upsample = 'DECONV' # upsample method in SegResNet
val_interval = 1
workers = 8




backward_unfreeze=True
CV_flag = True
DDP = False
exp_ensemble = False #'expert' ensemble
fix_samp_num_exp=False
freeze_specific=False #set false unless you want to train only one layer
freeze_train=True # start freezing layers in training if freeze criteria met
fs_ensemble = False# fewshot ensemble
isolate_layer=False #whether to isolate one layer at a time while freeze training
lr_cycling=True
super_val=False
train_partial = False
VAL_AMP = True





################EVAL SPECIFIC VARIABLES#################
eval_path = '/scratch/a.bip5/BraTS/BraTS21_data/'+'2023-09-25T1695604583.2732697Cluster_2SegResNet' #for non cluster path
eval_folder ='/scratch/a.bip5/BraTS/weights/m1695938467'
eval_mode = 'cluster' #choose between ensemble, cluster, online_val.
output_path ='/scratch/a.bip5/BraTS/saved_predictions'
test_samples_from='test'#'trainval' evalmode will test performance on training+val

slice_dice=True
plot_output=False
# unused_collection=True if fs_ensemble==0 else False