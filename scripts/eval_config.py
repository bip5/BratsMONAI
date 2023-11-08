import monai
import torch
model='SegResNet' # UNet
max_samples=150 #max number of samples to use for testing
load_save=0 # whether to load a saved model
load_name="./2022-01-20T16best_metric_model.pth" #Path to load saved model
batch_size=1 
save_name=f'{model}'
upsample='DECONV' # which upsampling to use for SegResNet choose deconv options- NONTRAINABLE, DECONV, PIXELSHUFFLE")
ensemble=0 # 0 for no 1 for yes to ensemble
avgmodel=0 # switch to average model 1=yes
plot=2 # only applicable in ensemble, 1= ensemble and plot, 2 evaluate all models
data_dir='/scratch/a.bip5/BraTS 2021/BraTS21_data'
eval_path='/scratch/a.bip5/BraTS 2021/BraTS21_data/'+'2023-09-07T1694090754.0773337Cluster_3SegResNet' #ssensemblemodels0922/Evaluation Folder C' # where to load the eval models for ensemble from
csv_name=f'{model}' # option to add additional details to csv name
workers=8 # how many cores to use
reduction='none' # reduction for key metric in evaluate ensemble, mean, none etc
val=10 # whether or not to only use the training set, set to 10 for analyse all data, 
schedule_epochs = 0
lr=1e-4
weight_decay=1e-5
val_interval = 1
VAL_AMP = True
output_images_dir='/scratch/a.bip5/BraTS 2021/ssensemblemodels0922/outputs'
loss_function = monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer_name ='Adam'
lr_scheduler_name = 'CosineAnnealingLR'
saver_ori = monai.transforms.SaveImage(output_dir=output_images_dir, output_ext=".nii.gz", output_postfix="ori",print_log=True)
saver_gt = monai.transforms.SaveImage(output_dir=output_images_dir, output_ext=".nii.gz", output_postfix="gt",print_log=True)
saver_seg = monai.transforms.SaveImage(output_dir=output_images_dir, output_ext=".nii.gz", output_postfix="seg",print_log=True)
dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = monai.metrics.DiceMetric(include_background=True, reduction="mean_batch")
save_output=0
# to only analyse the dataset without considering performance