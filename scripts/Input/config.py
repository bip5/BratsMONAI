root_dir="/scratch/a.bip5/BraTS 2021/"
model_name='SegResNet' # pick from UNet SegResNet
total_epochs=1200
val_interval = 1
VAL_AMP = True
load_save=0
seed=0
load_path='ENTER A PATH'
fold_num=1
max_samples=10
method='A'#type of ensemble method
upsample='DECONV' # upsample method in SegResNet
batch_size=4
T_max=4 #how often to reset cycling
CV_flag=1
lr=2e-4