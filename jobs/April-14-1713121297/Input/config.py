batch_size = 4
cluster_files = '/scratch/a.bip5/BraTS/cl4_train_2024-04-11_15-00-48.xlsx'#cl4_train_2024-04-04_14-08-47.xlsx'# '/scratch/a.bip5/BraTS/cl4_train_2024-01-17_15-40-25.xlsx' #cl4_B23al_2023-11-21_16-20-10.xlsx'#cl4_merge_2023-12-18_14-41-48.xlsx'# 
dropout=0
exp_train_count=5
exp_val_count=5
freeze_patience=2
fold_num = 1
init_filter_number= 32#16 # So we don't have keep creating new networks
#set depending on how many items to unfreeze from layer list
load_save = 1
load_path ='/scratch/a.bip5/BraTS/weights/job_7801275/SegResNetCV1_j7801275ep87'#'/scratch/a.bip5/BraTS/weights/m2023-11-07_20-07-54/SegResNetCV1_j7688198ep128'# '/scratch/a.bip5/BraTS/weights/job_7800978/SegResNetCV1_j7800978ep124'# '/scratch/a.bip5/BraTS/weights/job_7766530/2024-02-05SegResNet_half_j7766530'# #'/scratch/a.bip5/BraTS/weights/m2023-11-13_23-07-03/transformerCV1_j7692226ep82' #'/scratch/a.bip5/BraTS/weights/m2023-11-07_20-07-54/SegResNetCV1_j7688198ep77'#'/scratch#'/scratch/a.bip5/BraTS/weights/m2023-12-11_21-59-50/2023-12-12SegResNet_CA_half_j7716246'#
lr = 0.0002
root_dir = "/scratch/a.bip5/BraTS/BraTS_23_training/" # /scratch/a.bip5/ATLAS_2/Training #
weights_dir = "/scratch/a.bip5/BraTS/weights"
max_samples = 1250
method = 'A'#type of ensemble method
model_name = 'SegResNet'# 'SegResNet_half'#'SegResNetVAE'#'ScaleFocus' #'DualFocus'# 'WNet'#'UNet'#'transformer' #'DeepFocus'# pick from UNet SegResNet transformer #change roi when changing model name 
num_layers=2 # only used for custom models
num_filters=64 #only for deep focus line of models
plots_dir='/scratch/a.bip5/BraTS/plots'
PRUNE_PERCENTAGE= None # -0.05 #
roi=[192,192,144]#[192,192,128]#[128,128,128]#
seed = 0
total_epochs = 20
T_max = total_epochs #how often to reset cycling
unfreeze = 22  # only for freeze variants
upsample = 'DECONV' #'NONTRAINABLE'# upsample method in SegResNet
val_interval = 1
workers = 4
dropout=None

# 0 or 1: 0 will place all temporal samples inside test set and remaining in validation, 1 will place 50 pairs in training, 29 pairs in val and 29 pairs in test
temporal_split = 0
raw_features_filename= 'trainFeatures_5x' 






mode_index = 1

if mode_index==0:
    training_mode='CV_fold'
elif mode_index==1:
    training_mode='exp_ensemble'
elif mode_index==2:
    training_mode='val_exp_ens'
elif mode_index==3:
    training_mode='fs_ensemble'
elif mode_index==4:
    training_mode='CustomActivation'
    model_name='SegResNet_CA'
elif mode_index==5:
    training_mode='Flipper'
    model_name='SegResNet_Flipper'
elif mode_index==6:
    training_mode='CustomActivation'
    model_name='SegResNet_CA2'
elif mode_index==7:
    training_mode='CustomActivation'
    model_name='SegResNet_CA_half'
elif mode_index==8:
    training_mode='CV_fold'
    model_name='SegResNet_half'
elif mode_index==9:
    training_mode='CustomActivation'
    model_name='SegResNet_CA_quarter'
elif mode_index==10:
    training_mode='CustomActivation'
    model_name='SegResNet_CA2_half'
elif mode_index==11:
    training_mode='SegResNetAtt'
    model_name='SegResNetAtt'
    init_filter_number=init_filter_number/2 #Just to ease work flow in switching 
elif mode_index==12:
    training_mode='LoadNet'
    model_name='SegResNet_half'
    init_filter_number=16 #Just to ease work flow in switching 
else: 
    raise Exception('Invalid mode index please choose an appropriate value')
    
backward_unfreeze=False
binsemble=False
CV_flag = False
DDP = False
exp_ensemble = True #'expert' ensemble
fix_samp_num_exp= False # True # 
freeze_specific = False #set false unless you want to train only one layer
freeze_train = False # start freezing layers in training if freeze criteria met
fs_ensemble = False # fewshot ensemble
isolate_layer = False #whether to isolate one layer at a time while freeze training
lr_cycling = True
super_val = False
train_partial = False
VAL_AMP = False
DE_option='plain'#'squared'#
TTA_ensemble=True



################EVAL SPECIFIC VARIABLES#################
eval_path = '/scratch/a.bip5/BraTS/weights/m2023-11-07_20-07-54/SegResNetCV1_j7688198ep128' #only used when evaluating a single model
eval_folder ='/scratch/a.bip5/BraTS/weights/job_7802613/' #'/scratch/a.bip5/BraTS/weights/m2024-01-22_19-23-02' #'/scratch/a.bip5/BraTS/weights/m2023-11-07_20-07-54'#'/scratch/a.bip5/BraTS/weights/m2023-11-22_18-52-42'# tfmr #'/scratch/a.bip5/BraTS/weights/m2023-11-22_00-04-01'#
eval_mode =  'cluster_expert' #'distance_ensemble'# 'time' #'simple'#    'cluster'  # 'online_val'# choose between simple, cluster, online_val. 
output_path = '/scratch/a.bip5/BraTS/saved_predictions'
test_samples_from ='trainval'#'test'# # evalmode will test performance on training+val ###CHECK
plot_list = None #['00619-001','01479-000','00113-000','01498-000','01487-000','0025-000','01486-000','01327-000','0684-000','0155-000','01433-000','0084-001','0753-000','0012-000','01169-000','01483-000','00152-000'] #     #use if you want to plot specific samples
limit_samples = None # 10 #  only evaluate limited samples

use_cluster_for_online_val = False 
slice_dice=True
plot_output= False #True # 
plot_single_slice=False
eval_from_folder=  True #False #
# unused_collection=True if fs_ensemble==0 else False