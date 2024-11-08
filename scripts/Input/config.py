load_path ='/scratch/a.bip5/BraTS/weights/job_7808186/SegResNetCV1_j7808186ep90'
encoder_path = load_path#'/scratch/a.bip5/BraTS/weights/job_7807898/SegResNetCV1_j7807898ep93' # repsplit#
load_save = 1
jit_model =False
batch_size = 4
cluster_files ='/scratch/a.bip5/BraTS/cluster_files/cl4_Tfmrft_04July_ts1.xlsx' #'/scratch/a.bip5/BraTS/cl4_tf_18thApril_repSplit.xlsx'#cl4_trainFeatures_5x_repSplit.xlsx'#cl4_ft_25April_ts0.xlsx'# cl4_train_2024-04-04_14-08-47.xlsx'# '/scratch/a.bip5/BraTS/cl4_train_2024-01-17_15-40-25.xlsx' #cl4_B23al_2023-11-21_16-20-10.xlsx'#cl4_merge_2023-12-18_14-41-48.xlsx'# 
dropout=0
loss_type = 'dice'# 'EdgyDice'# 'InvDice'#'CE'#'lesionwiseVal'#'lesion_wise'#'MaskedDiceLoss'#'whatever'#
exp_train_count=5
exp_val_count=5
freeze_patience=2
fold_num = 5
init_filter_number= 32#16 # So we don't have keep creating new networks
#set depending on how many items to unfreeze from layer list

#'/scratch/a.bip5/BraTS/weights/job_7807898/SegResNetCV1_j7807898ep93'#representative subject separation# '/scratch/a.bip5/BraTS/weights/job_7808186/SegResNetCV1_j7808186ep90' #all temporal in test set# '/scratch/a.bip5/BraTS/weights/m2023-11-07_20-07-54/SegResNetCV1_j7688198ep128'# sample only separation # 
lr = 0.0002
root_dir = "/scratch/a.bip5/BraTS/BraTS_23_training/" # /scratch/a.bip5/ATLAS_2/Training #
weights_dir = "/scratch/a.bip5/BraTS/weights"
max_samples = 1250
# method = 'A'#type of ensemble method - where is this used? disabled since not used pending deletion
model_name = 'SegResNet'# 'SegResNet_half'#'SegResNetVAE'#'ScaleFocus' #'DualFocus'# 'WNet'#'UNet'#'transformer' #'DeepFocus'# pick from UNet SegResNet transformer #change roi when changing model name 
num_layers = 2 # only used for custom models
num_filters = 64 #only for deep focus line of models
plots_dir='/scratch/a.bip5/BraTS/plots'
PRUNE_PERCENTAGE= None # -0.05 #
roi=[192,192,144]#[192,192,128]#[128,128,128]#
seed = 0
dataset_seed= 0
total_epochs = 150
T_max = total_epochs #how often to reset cycling
unfreeze = 22  # only for freeze variants
upsample = 'DECONV' #'NONTRAINABLE'# upsample method in SegResNet
val_interval = 1
workers = 4
inf_overlap=0.5
base_transform_probability = 0.3
dropout=None
use_sampler =  False#
minival=False#
no_val=False
xval=False
checkpoint_snaps=False
load_base = False # here to avoid import errors
base_path = None # here to avoid import errors
incremental_transform = True
skip_AMP =True

activation= 'RELU'# 'hardswish' #  # here to avoid import errors
in_channels=4
out_channels=3
# 0 or 1: 0 will place all temporal samples inside test set and remaining in validation, 1 will place 50 pairs in training, 29 pairs in val and 29 pairs in test
temporal_split = 0
if temporal_split == 1:
    val_temporal = 10 #29
    test_temporal = 14 #29
    train_temporal = 94 #60 /should sum  to 118
else:
    val_temporal,test_temporal,train_temporal = None,None,None
raw_features_filename= 'ft_25April_ts0'# 'TfmrBotN_ft_13July'#'Tfmrft_04July'# 'tf_18thApril_repSplit' # 'trainFeatures_5x_repSplit'# #Ensure correct file for evaluation.

training_samples=1000




mode_index =19
print('MODE INDEX ',mode_index)

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
elif mode_index==13:
    training_mode= 'Infusion'
elif mode_index==14:
    no_val=False
    load_path = '/scratch/a.bip5/BraTS/weights/job_7878869/transformerCV1_j7878869ep115_ts1'
    encoder_path = load_path
    load_save=1
    training_mode= 'exp_ensemble'#'val_exp_ens' #
    model_name = 'transformer'
    lr = lr*4
    total_epochs = 20
    loss_type='dice' #'EdgyDice'
    roi = [128,128,128]
    batch_size = 1
    inf_overlap= 0.7
    use_sampler =  False#
    minival=False
elif mode_index==15:
    no_val = True
    lr=lr/10
    total_epochs=150
    max_samples=1251
    training_mode='CV_fold'
elif mode_index==16:
    load_save=0
    
    no_val = False
    xval = False
    total_epochs=150
    max_samples=1250
    training_mode='intersect'
elif mode_index==17:
    load_save=1
    load_base = True
    base_path = '/scratch/a.bip5/BraTS/weights/job_7808186/SegResNetCV1_j7808186ep90'
    training_mode='ClusterBlend'
    model_name='ClusterBlend'
elif mode_index==18:
    load_save=1
    load_base = True
    base_path = '/scratch/a.bip5/BraTS/weights/job_7808186/SegResNetCV1_j7808186ep90'
    training_mode='PixelLayer'
    model_name='PixelLayer'
    activation= 'hardswish' #'RELU'# 
elif mode_index==19:
    load_save = 0
    
    load_path = '/scratch/a.bip5/BraTS/weights/job_7957665/2024-11-05SegResNetDS_j7957665_ts0_LL'#'/scratch/a.bip5/BraTS/weights/job_7957288/2024-11-04SegResNetDS_j7957288_ts0'#'/scratch/a.bip5/BraTS/weights/0_NVAUTO_models/model14.ts'  # '/scratch/a.bip5/BraTS/weights/job_7956992/2024-11-02SegResNetDS_j7956992_ts0' #'/scratch/a.bip5/BraTS/weights/job_7953765/2024-10-20SegResNetDS_j7953765_ts0' #
    jit_model= False
    root_dir = '/scratch/a.bip5/BraTS/dataset-ISLES22^public^unzipped^version' 
    model_name = 'SegResNetDS'
    training_mode = 'isles'
    max_samples = 250
    roi = (192,192,128)
    total_epochs = 1000
    init_filter_number= 32
    batch_size=4
    in_channels = 2
    out_channels = 2    
    inf_overlap = 0.7
    lr = 0.0002
    activation = 'RELU'
    loss_type = 'DiceFocal' #'dice'#
    seed = 1
    dataset_seed = 8
    incremental_transform = False
    if incremental_transform:
        if load_save==0:
            training_samples = 200
        else:
            training_samples = 200
    else:
        training_samples = 200
    base_transform_probability=1 if incremental_transform else 0.3
    skip_AMP =True
elif mode_index==20:
    
    load_save = 0
    load_path = '/scratch/a.bip5/BraTS/weights/job_7953042/2024-10-18SegResNet_j7953042_ts0_LL'  
    root_dir = '/scratch/a.bip5/ATLAS_2/Training/' 
    model_name = 'SegResNet'
    batch_size=4
    training_mode = 'atlas'
    max_samples = 655
    loss_type = 'dice'#'DiceFocal'#
    roi = (192,192,144)#(64,64,64)#(128,128,128)#
    total_epochs = 100
    init_filter_number= 32
    lr = 0.000025
    in_channels = 1
    out_channels = 1
    inf_overlap = 0.8   
    seed = 11
    dataset_seed = 1
    incremental_transform = False
    training_samples= 100 if incremental_transform else 600
    # workers=1
elif mode_index==21:
    
    load_save = 0
    load_path = '/scratch/a.bip5/BraTS/weights/job_7953042/2024-10-18SegResNet_j7953042_ts0_LL'  
    root_dir = "/scratch/a.bip5/BraTS/BraTS_23_training/" 
    model_name = 'SegResNetDS'
    batch_size=4
    training_mode = 'pretrain'
    max_samples = 1250
    loss_type = 'DiceFocal'#'dice'
    roi = (192,192,144)#(64,64,64)#(128,128,128)#
    total_epochs = 100
    init_filter_number= 32
    in_channels = 2
    out_channels = 2
    inf_overlap = 0.8   
    seed = 11
    dataset_seed = 1
    incremental_transform = False
    training_samples= 100 if incremental_transform else 600
    # workers=1
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
VAL_AMP = False #True
DE_option='plain'#'squared'#
TTA_ensemble=False



################EVAL SPECIFIC VARIABLES#################
eval_mode =   'simple'# 'jit ens'#'distance_ensemble'#  'cluster_expert' #    'time' # 'online_val'#    'cluster'  #  choose between simple, cluster, online_val. 
eval_path = load_path #only used when evaluating a single model
base_perf_path = '/scratch/a.bip5/BraTS/jobs_eval/May-29-1716998494/IndScoresjob_7878869_7885546.xlsx'#TransformerRepSplitBase#'/scratch/a.bip5/BraTS/jobs_eval/April-18-1713463104/IndScores_7809063.xlsx' #Repsplitbase perf#'/scratch/a.bip5/BraTS/jobs_eval/April-25-1714065419/IndScoresjob_7808186_7860766.xlsx'# 
eval_folder = '/scratch/a.bip5/BraTS/weights/0_NVAUTO_models'

output_path = '/scratch/a.bip5/BraTS/saved_predictions'
test_samples_from ='test'# 'trainval'## evalmode will test performance on training+val ###CHECK
plot_list = None #['00619-001','01479-000','00113-000','01498-000','01487-000','0025-000','01486-000','01327-000','0684-000','0155-000','01433-000','0084-001','0753-000','0012-000','01169-000','01483-000','00152-000'] #     #use if you want to plot specific samples
limit_samples = 10 #None # 10 #  only evaluate limited samples

online_val_mode = 'ensemble' # 'single'#'cluster' # 
slice_dice=False
plot_output= False #True # 
plot_single_slice=False
eval_from_folder=  True #False #
# unused_collection=True if fs_ensemble==0 else False
