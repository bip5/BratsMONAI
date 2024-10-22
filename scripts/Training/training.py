import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/') # need to add the import folder to system path for python to know where to look for

import numpy as np
import os
import torch
from monai.data import DataLoader
from Input.localtransforms import (
train_transform,train_transform_isles,
train_transform_infuse,
train_transform_PA,
train_transform_CA,
train_transform_Flipper,
val_transform,val_transform_isles,
val_transform_PA,
val_transform_Flipper,
post_trans,
train_transform_atlas,
val_transform_atlas,
update_transforms_for_epoch,
isles_list
)

from Training.prun import prune_network
from monai.handlers.utils import from_engine
import torch.nn.functional as F
from Input.config import (
root_dir,
weights_dir,
total_epochs,
val_interval,
VAL_AMP,
fold_num,
seed,
batch_size,
T_max,
lr,
max_samples,
model_name,
workers,
load_save,
load_path,
DDP,
training_mode,
train_partial,
unfreeze,
freeze_train,
cluster_files,
lr_cycling,
isolate_layer,
super_val,
exp_train_count,
exp_val_count,
fix_samp_num_exp,
freeze_patience,
freeze_specific,
backward_unfreeze,
binsemble,
roi,
PRUNE_PERCENTAGE,
temporal_split,
loss_type,
use_sampler,
minival,
no_val,
checkpoint_snaps,
load_base,
base_path,
in_channels,
incremental_transform,
training_samples
)

from Training.loss_function import loss_function,edgy_dice_loss
from Training.network import model 
from Training.running_log import log_run_details
from Training.clusterBlend import ClusterBlend
from Training.pixelLayer import PixelLayer
from Evaluation.eval_functions import model_loader, model_loader_ind

from Training.optimiser import get_optimiser

from Evaluation.evaluation import (
dice_metric,
dice_metric_batch,
inference,
)

from Input.dataset import (
BratsDataset,
IslesDataset,
BratsInfusionDataset,
EnsembleDataset,
ExpDataset,
train_indices,
val_indices,
test_indices,
ClusterAugment,
AtlasDataset,
indexes
)

from monai.utils import set_determinism
from monai.data import decollate_batch,partition_dataset
from datetime import date, datetime
import time
from torchsummary import summary
from torch.utils.data import Subset,SubsetRandomSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd
import gc
from BraTS2023Metrics.metrics import get_LesionWiseResults as lesion_wise
from BraTS2023Metrics.metrics import LesionWiseDice
import matplotlib.pyplot as plt
from pathlib import Path
import wandb
# import psutil
# import threading


# def print_memory():
    # print(f'Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB')
    # print(f'Reserved:    {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB')
    # process = psutil.Process(os.getpid())
    # print(f'CPU memory : {process.memory_info().rss / 1024 ** 2:.2f} MB')

# def schedule_prints(interval=10):  # 10 seconds by default
    # threading.Timer(interval, schedule_prints, [interval]).start()
    # print_memory()

##Call this once before starting your training
# schedule_prints()

# Get a dictionary of the current global namespace
namespace = locals().copy()
# print('train_indices,val_indices,test_indices',train_indices,val_indices,test_indices)
# sys.exit()
config_dict=dict()
job_id = os.environ.get('SLURM_JOB_ID', 'N/A')
config_dict['job_id']=job_id
for name, value in sorted(namespace.items()):
    if not name.startswith("__"):
        if type(value) in [str,int,float,bool]:
            print(f"{name}: {value}")
            config_dict[f"{name}"]=value
wandb.init(
    # set the wandb project where this run will be logged
    project="segmentation",
    name = job_id,
    notes = os.environ.get('NOTE_FOR_WANDB', 'N/A'),
    # track hyperparameters and run metadata
    config=config_dict
)

current_file = Path(__file__).resolve() # This gets the root path two directories up

# Join with new path
config_path = current_file.parents[1] / 'Input' / 'config.py'

with open(config_path, 'r') as config_file:
    script_content = config_file.read()
print("\n\n------ Config Content ------\n")
print(script_content)
print("\n---------------------------\n")

os.environ['PYTHONHASHSEED']=str(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
set_determinism(seed=seed)
device = torch.device(f"cuda:0")
training_layers=["module.convInit.conv.weight", "module.down_layers.0.1.conv1.conv.weight", "module.down_layers.0.1.conv2.conv.weight", "module.down_layers.1.0.conv.weight", "module.down_layers.1.1.conv1.conv.weight", "module.down_layers.1.1.conv2.conv.weight", "module.down_layers.1.2.conv1.conv.weight", "module.down_layers.1.2.conv2.conv.weight", "module.down_layers.2.0.conv.weight", "module.down_layers.2.1.conv1.conv.weight", "module.down_layers.2.1.conv2.conv.weight", "module.down_layers.2.2.conv1.conv.weight", "module.down_layers.2.2.conv2.conv.weight", "module.down_layers.3.0.conv.weight", "module.down_layers.3.1.conv1.conv.weight", "module.down_layers.3.1.conv2.conv.weight", "module.down_layers.3.2.conv1.conv.weight", "module.down_layers.3.2.conv2.conv.weight", "module.down_layers.3.3.conv1.conv.weight", "module.down_layers.3.3.conv2.conv.weight", "module.down_layers.3.4.conv1.conv.weight", "module.down_layers.3.4.conv2.conv.weight", "module.up_layers.0.0.conv1.conv.weight", "module.up_layers.0.0.conv2.conv.weight", "module.up_layers.1.0.conv1.conv.weight", "module.up_layers.1.0.conv2.conv.weight", "module.up_layers.2.0.conv1.conv.weight", "module.up_layers.2.0.conv2.conv.weight", "module.up_samples.0.0.conv.weight", "module.up_samples.0.1.deconv.weight", "module.up_samples.0.1.deconv.bias", "module.up_samples.1.0.conv.weight", "module.up_samples.1.1.deconv.weight", "module.up_samples.1.1.deconv.bias", "module.up_samples.2.0.conv.weight", "module.up_samples.2.1.deconv.weight", "module.up_samples.2.1.deconv.bias", "module.conv_final.2.conv.weight", "module.conv_final.2.conv.bias"]

unfreeze_layers=training_layers[unfreeze:]
model_names=set() #to store unique model_names saved by the script
best_metrics=set()
once=0

def generate_pseudo_mask(predictions, target_dice):
    # Step 1: Create a binary mask  
    
    binary_mask = (predictions > 0.5 ).float()

    # Step 2: Calculate necessary reduction in positive pixels
    current_positive = binary_mask.sum()
    target_positive = target_dice * current_positive

    # Step 3: Remove the least certain pixels first
    # Calculate how far each pixel is from the threshold
    certainty = torch.abs(predictions - 0.5)

    # Sort by certainty and remove the least certain pixels
    _, indices = torch.sort(certainty.flatten())
    flat_binary_mask = binary_mask.flatten()
    flat_binary_mask[indices[:int(current_positive - target_positive)]] = 0

    # Reshape back to original shape
    sparse_mask = flat_binary_mask.reshape(binary_mask.shape)

    return sparse_mask

def generate_random_patch_masks(batch_size, num_output_channels, roi,patch_size=0.5, density=1.0):
    """
    Generate a random patch mask for each sample in a batch.

    Parameters:
    - batch_size (int): The number of samples per batch.
    - num_output_channels (int): The number of output channels.
    - height (int): The height of the output image.
    - width (int): The width of the output image.
    - patch_size (tuple of int): A tuple (patch_height, patch_width) defining the size of the patch.
    - density (float): The density of 1's in the patch (default 1.0, fully filled).

    Returns:
    - numpy.ndarray: A batch of random patch masks of shape (batch_size, num_output_channels, height, width).
    """
    height, width,depth = roi
    masks = np.zeros((batch_size, num_output_channels, height, width,depth), dtype=np.uint8)
    patch_height, patch_width, patch_depth  = [int(patch_size* float(x)) for x in roi]

    for i in range(batch_size):
        for j in range(num_output_channels):
            # Ensure the patch fits within the image dimensions
            start_x = np.random.randint(0, height - patch_height + 1)
            start_y = np.random.randint(0, width - patch_width + 1)
            start_z = np.random.randint(0, depth - patch_depth + 1)
            # Fill the patch area with 1's based on the specified density
            if density == 1.0:
                masks[i, j, start_x:start_x + patch_height, start_y:start_y + patch_width, start_z:start_z + patch_depth] = 1
            else:
                masks[i, j, start_x:start_x + patch_height, start_y:start_y + patch_width] = np.random.choice(
                    [0, 1], size=(patch_height, patch_width), p=[1-density, density])

    return masks
    
def generate_random_masks(batch_size, num_output_channels, height, width, depth,density=0.5):
    """
    Generate a random binary mask for each sample in a batch.

    Parameters:
    - batch_size (int): The number of samples per batch.
    - num_output_channels (int): The number of output channels (depth of mask).
    - height (int): The height of the output image.
    - width (int): The width of the output image.
    - density (float): The probability of a mask element being 1 (default 0.5).

    Returns:
    - numpy.ndarray: A batch of random binary masks of shape (batch_size, num_output_channels, height, width).
    """
    return np.random.choice([0, 1], size=(batch_size, num_output_channels, height, width,depth), p=[1-density, density])

def check_gainless_epochs(epoch, best_metric_epoch):
    gainless_epochs = epoch  - best_metric_epoch
    if gainless_epochs >= freeze_patience:
        return True
    else:
        return False

now = datetime.now()
formatted_time =now.strftime('%Y-%m-%d_%H-%M-%S')

total_start = time.time()
save_dir=os.path.join(weights_dir,'job_'+str(job_id))

os.makedirs(save_dir,exist_ok=True)
print('SAVING MODELS IN ', save_dir)
best_metric_epoch=0



# torch.cuda.set_device(device)
model=model.to(device)  
# model=torch.nn.DataParallel(model)


with torch.cuda.amp.autocast():
    print(training_mode)
    if training_mode=='Flipper':            
        summary(model,(8,192,192,128)) 
    else:
        summary(model,(in_channels,*roi))
   
        
torch.manual_seed(seed)    
# model=DistributedDataParallel(module=model, device_ids=[device],find_unused_parameters=False)

    

optimiser = get_optimiser(model)

scaler = torch.cuda.amp.GradScaler()
print("Model defined and passed to GPU")
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True



lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max) #torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max) 
start_epoch=0
if load_save==1:
    model.train()
    if training_mode=='CustomActivation':
        saved_state_dict = torch.load(load_path)
        new_state_dict = model.state_dict()
        
        for name, param in new_state_dict.items():
            if name in saved_state_dict:
                param.data.copy_(saved_state_dict[name].data)
            else:
                print(f"Skipping {name}, not present in the saved state dict.")
                    
        # for name, param in model.named_parameters():
            # if 'act_bias' in name or 'act_scale' in name:
                # param.requires_grad = True  # Allow training only on new activation parameters
            # else:
                # param.requires_grad = False  # Freeze all other parameters
                
    
    elif training_mode=='ClusterBlend':
        if load_base:
            base_model = model_loader(base_path)
            model=ClusterBlend(base_model).to(device)
        else:
            model = ClusterBlend(model).to(device)
            model = model_loader_ind(model)
            
    elif training_mode=='PixelLayer':
        if load_base:
            base_model = model_loader(base_path)
            model=PixelLayer(base_model).to(device)
        else:
            model = PixelLayer(model).to(device)
            model = model_loader_ind(model)
    
    else: 
        try:
            
            model,optimiser,scaler,lr_scheduler,start_epoch = model_loader(load_path,train=True,scaler=scaler,lr_scheduler=lr_scheduler)
           
            print('LOADED STATE SUCCESSFULLY')
        except:
            model = model_loader(load_path,train=True)
        
        print("loaded saved model ", load_path)
        if PRUNE_PERCENTAGE is not None:
            model = prune_network(model)
            print('PRUNED MODEL!?!')
        if train_partial==True:
            # Step 2: Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
            
            for name, param in model.named_parameters():
                if name in unfreeze_layers:
                    param.requires_grad = True
            print(f'only training {unfreeze_layers}')
else:
    if training_mode=='ClusterBlend':
        model = ClusterBlend(model).to(device)
    elif training_mode=='PixelLayer':
        model = PixelLayer(model).to(device)
    
def validate(val_loader, epoch, best_metric, best_metric_epoch, sheet_name=None, save_name=None, custom_inference=None):
    def default_inference(val_data):
        val_inputs = val_data["image"].to(device)
        val_masks = val_data["mask"].to(device)
        
        val_data["pred"] = inference(val_inputs, model)
        
        val_data = [post_trans(i) for i in decollate_batch(val_data)]
        val_outputs, val_masks = from_engine(["pred", "mask"])(val_data)
        
        return val_outputs, val_masks
    
    model.eval()
    alt_metrics = []
   
    
    #picks default when custom is none
    run_inference = custom_inference or default_inference

    
    with torch.no_grad():
        for val_data in val_loader:
            val_outputs, val_masks = run_inference(val_data)
            
            # Consistent dice calculation
            val_outputs = [tensor.to(device) for tensor in val_outputs]
            val_masks = [tensor.to(device) for tensor in val_masks]
            
            dice_metric(y_pred=val_outputs, y=val_masks)
            dice_metric_batch(y_pred=val_outputs, y=val_masks)
            
            if loss_type == 'EdgyDice':            
                alt_metric = LesionWiseDice(val_outputs, val_masks)
                alt_metrics.append(alt_metric)
    
    if loss_type == 'EdgyDice':            
        metric = np.mean(alt_metrics)
    else:
        metric = dice_metric.aggregate().item()
    
    modes = ['isles', 'atlas']
    if training_mode not in modes:      
        metric_batch = dice_metric_batch.aggregate()
        metric_tc = metric_batch[0].item()
        metric_wt = metric_batch[1].item()
        metric_et = metric_batch[2].item()
    
    dice_metric.reset()
    dice_metric_batch.reset()

    if metric > best_metric:
        best_metric = metric
        best_metric_epoch = epoch + 1
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimiser.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
        }
        
        if training_mode == 'CV_fold':   
            save_name = f"{model_name}CV{fold_num}_j{job_id}{'ep'+str(epoch+1) if checkpoint_snaps else ''}_ts{temporal_split}"
        elif sheet_name is not None:  
            save_name = f"{sheet_name}_{load_save}_j{job_id}{'_e'+str(best_metric_epoch) if checkpoint_snaps else ''}"
        else:
            save_name = f"{date.today().isoformat()}{model_name}_j{job_id}_ts{temporal_split}"
        
        saved_model = os.path.join(save_dir, save_name)
        torch.save(state, saved_model)
        print(f"Saved new best metric model: {saved_model}")
    
    if training_mode in modes:
        print(
            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"            
            f"\nbest mean dice: {best_metric:.4f}"
            f" at epoch: {best_metric_epoch}"
        )
    else:
        print(
            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
            f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
            f"\nbest mean dice: {best_metric:.4f}"
            f" at epoch: {best_metric_epoch}"
        )
    
    return save_name, best_metric, best_metric_epoch, metric


    
def trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir,model=model,sheet_name=None,once=once,**kwargs):
    last_model=None
    print("number of files processed: ", train_dataset.__len__()) #this is not
    
  
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=workers )
    print('loading val data')
    if training_mode=='val_exp_ens':
        val_loader0=DataLoader(val_dataset[0], batch_size=batch_size, shuffle=False,num_workers=workers)
        val_loader1=DataLoader(val_dataset[1], batch_size=batch_size, shuffle=False,num_workers=workers)
        val_loader2=DataLoader(val_dataset[2], batch_size=batch_size, shuffle=False,num_workers=workers)
        val_loader3=DataLoader(val_dataset[3], batch_size=batch_size, shuffle=False,num_workers=workers)
    else:
        val_loader=DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=workers)
    print("All Datasets assigned")                                             

    

        
    
    


    # use amp to accelerate training

    
    best_metric = -1
    best_loss = 1
    best_metric_epoch = -1
    # best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    if training_mode=='val_exp_ens':
        cluster_names=['Cluster_0','Cluster_1','Cluster_2','Cluster_3']
        best_metric=[-1]*len(cluster_names)
        best_metric_epoch=[-1]*len(cluster_names)
        metric = [-1]*len(cluster_names)




   
    print("starting epochs")
    gainless_counter=0 #to check how many times gainless function has tripped, resets after all layers thawed at least once
    patience_counter=0 #separate counter which resets at checkpoint
    saved_models_count = 0
    last_saved_models_count=0
    model_performance_dict = {}  # to store {model_path: dice_score}
    best_met_old=0
    train_dice_scores = [] # 1- epoch loss
    val_scores = []
    new_samples= len(train_indices)
    for epoch in range(start_epoch,total_epochs):
        print_ids=0
        if incremental_transform:
            if training_mode=='isles':
               
                if epoch==best_metric_epoch:
                    if new_samples<230:                        
                        new_samples = new_samples+50
                        new_indices=indexes[:new_samples]
                        print(new_indices)
                        full_train=IslesDataset("/scratch/a.bip5/BraTS/dataset-ISLES22^public^unzipped^version"  ,transform= train_transform_isles )
                        train_dataset = Subset(full_train, new_indices)   
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers ) 
                        print('INTRODUCED NEW SAMPLES')
                        print_ids=1
                    else:
                        print('AUGMENTATION UPDATE')
                        updated_transform_isles = update_transforms_for_epoch(isles_list,epoch,total_epochs)

                        full_train=IslesDataset("/scratch/a.bip5/BraTS/dataset-ISLES22^public^unzipped^version"  ,transform= updated_transform_isles )
                        train_dataset = Subset(full_train, new_indices)   
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=workers ) 
                   
                # elif (epoch-best_metric_epoch)%10==0:
                    # if new_samples<230:
                        
                        # new_samples = new_samples+10
                        # new_indices=indexes[:new_samples]
                        # print(new_indices)
                        # full_train=IslesDataset("/scratch/a.bip5/BraTS/dataset-ISLES22^public^unzipped^version"  ,transform= train_transform_isles )
                        # train_dataset = Subset(full_train, new_indices)   
                        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers ) 
                        # print('INTRODUCED NEW SAMPLES')
                        # print_ids=1
                        
            if training_mode=='atlas':
                if epoch==best_metric_epoch:
                    if new_samples<600:                        
                        new_samples = new_samples+100
                        new_indices=indexes[:new_samples]
                        print(new_indices)
                        full_train=AtlasDataset(root_dir ,transform= train_transform_isles )
                        train_dataset = Subset(full_train, new_indices)   
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers ) 
                        print('INTRODUCED NEW SAMPLES')
                        print_ids=1
                    else:
                        print('AUGMENTATION UPDATE')
                        updated_transform_isles = update_transforms_for_epoch(isles_list,epoch,total_epochs)

                        full_train=AtlasDataset(root_dir ,transform= updated_transform_isles )
                        train_dataset = Subset(full_train, new_indices)   
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=workers ) 
                    
                # elif (epoch-best_metric_epoch)%5==0:
                    # if new_samples<600:
                        
                        # new_samples = new_samples+10
                        # new_indices=indexes[:new_samples]
                        # print(new_indices)
                        # full_train=AtlasDataset(root_dir ,transform= train_transform_isles )
                        # train_dataset = Subset(full_train, new_indices)   
                        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers ) 
                        # print('INTRODUCED NEW SAMPLES')
                        # print_ids=1
                    
                
        indices = list(range(1000))
        np.random.shuffle(indices)
        if use_sampler:
            # Define the size of the subset you want to use each epoch
            subset_size = 50  # Adjust this to whatever size you want

            # Create the sampler
            sampler = SubsetRandomSampler(indices[:subset_size])
           
            train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=workers, sampler=sampler)
                
            # train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{total_epochs}")
        
        epoch_loss = 0
        step = 0
        
        if freeze_train:             
            if freeze_specific:
                unfreeze_layers=training_layers[-4:]
                for param in model.parameters():
                    param.requires_grad = False
                for name, param in model.named_parameters():
                    if name in unfreeze_layers:
                        param.requires_grad = True
                print(f' only training layer {unfreeze_layers}, commencing training')
            elif backward_unfreeze:
                unfreeze_layers=training_layers[-(gainless_counter+1):]
                print(f'training {unfreeze_layers}')
                if epoch==0:
                    for param in model.parameters():
                        param.requires_grad = False
                    
                    for name, param in model.named_parameters():
                        if name in unfreeze_layers:
                            param.requires_grad = True
                if check_gainless_epochs(epoch, best_metric_epoch):
                    gainless_counter+=1 #to decide which layer to isolate
                    # unfreeze_index=len(training_layers)%gainless_counter -1
                    model.load_state_dict(torch.load(saved_model),strict=False)
                    print(f'loaded {saved_model} to commence training')
                    if (gainless_counter)==len(training_layers):
                        gainless_counter=0 # reset gainless counter
                    if len(unfreeze_layers)==len(training_layers):
                        print('Nothing left to freeze, this is probably as good as it gets, play with other hyps maybe?')
                        break
                    for param in model.parameters():
                        param.requires_grad = False
                    
                    for name, param in model.named_parameters():
                        if name in unfreeze_layers:
                            param.requires_grad = True
                    print(f' Unfroze {gainless_counter} items, commencing training')  
              
            # check gainless epoch only activates each patience cycle
            else: 
                if epoch==0:
                    unfreeze_layers=training_layers[-1]
                    for param in model.parameters():
                        param.requires_grad = False
                    
                    for name, param in model.named_parameters():
                        if name in unfreeze_layers:
                            param.requires_grad = True
                if check_gainless_epochs(epoch, best_metric_epoch):
                    gainless_counter+=1 #to decide which layer to isolate
                    best_metric_epoch=epoch #to reset gainless epoch check
                    unfreeze_index=len(training_layers)-gainless_counter 
                    print(unfreeze_index,'unfreeze_index')
                    model.load_state_dict(torch.load(saved_model),strict=False)
                    print(f'loaded {saved_model} to commence training')
                    
                    
                    if isolate_layer:
                        unfreeze_layers=training_layers[unfreeze_index]
                        print(f'training {unfreeze_layers}')
                        
                        if gainless_counter==len(training_layers):
                            gainless_counter=0 # reset gainless counter
                            
                            print('Nothing left to freeze, restarting gainless counter and training whole model for now')
                            
                        else:
                            for param in model.parameters():
                                param.requires_grad = False
                            for name, param in model.named_parameters():
                                if name in unfreeze_layers:
                                    param.requires_grad = True
                        
        
        
        model.train()
        for ix ,batch_data in enumerate(train_loader): 
            # if ix==13:
                # print('next epoch please')
                # break
            # print(ix, 'just printing to see what up')
            if print_ids:
                print(batch_data['id'])
            if epoch==0:
                if 'map_dict' in kwargs:
                    map_dict=kwargs['map_dict']
                    print(batch_data['id'])
                    for sid in batch_data['id']:
                        if training_mode=='exp_ensemble':
                            sample_index=map_dict[sid]
                            assert sample_index in train_indices , 'Training outside index'
                
            step_start = time.time()
            step += 1
            inputs, masks = (
                batch_data["image"].to(device),
                batch_data["mask"].to(device),
            )
            optimiser.zero_grad()
            
                
            with torch.cuda.amp.autocast():
                outputs = inference(inputs,model)#model(inputs)
                
                if loss_type == 'MaskedDiceLoss':
                    height,width,depth = roi
                    loss_mask = generate_random_patch_masks(1, 1, roi)
                    loss_mask2 = torch.from_numpy(loss_mask).to(device)
                    loss = loss_function(outputs, masks,loss_mask2) 
                elif loss_type == 'lesion_wise':
                    with torch.no_grad():
                        thresholded_output = (torch.sigmoid(outputs) > 0.5).float()
                        lesionwise_dice= 1 - LesionWiseDiceLoss(thresholded_output, masks)
                        print('lesionwise_dice',lesionwise_dice)
                        
                        new_target = generate_pseudo_mask(thresholded_output,lesionwise_dice)
                        new_target.to(device)
                    loss = loss_function(outputs, new_target) 
                elif model_name == 'SegResNetDS':
                    # outputs = model(inputs)
                    losses = []
                    weights = torch.tensor([1.0, 0.5, 0.25,0.125], requires_grad=True).to(device)
                    for i, output in enumerate(outputs):
                        # torch.Size([4, 1, 192, 192, 128]) output.shape
                        # torch.Size([4, 1, 96, 96, 64]) output.shape
                        # torch.Size([4, 1, 48, 48, 32]) output.shape
                        # torch.Size([4, 1, 24, 24, 16]) output.shape output shapes from DS
                        #print(output.shape,'output.shape')# 4 resolutions for each batch
                        #print(masks.shape)# masks.shape: ([4, 1, 192, 192, 128])
                        masks_resized=[]
                        mask=[]
                        for bnum in range(output.shape[0]):
                            # print(masks.shape, ' masks shape')
                            # print(output.shape, 'output shape')
                            mask = F.interpolate(masks[0,:,:,:,:].unsqueeze(0), size = output.shape[-3:], mode='nearest')
                            masks_resized.append(mask)
                        mask_resized = torch.cat(masks_resized,dim=1)
                        loss = loss_function(output.unsqueeze(0), mask_resized)
                        losses.append(loss * weights[i])
                    loss = sum(losses)
                                       
                else:
                    loss = loss_function(outputs, masks) 
                    # print(type(outputs))
                    # print(loss, 'original loss')
                    edgy_loss_calc = edgy_dice_loss(outputs, masks)
                    # print(edgy_loss_calc, 'calculated edgy dice loss')
                          
            scaler.scale(loss).backward()
            try:
                scaler.step(optimiser)
            except AssertionError as e:
                print(f"AssertionError caught: {e}")
            scaler.update()
            epoch_loss += loss.item()
            if step%10==0:
                print(
                    f"{step}/{len(train_dataset) // train_loader.batch_size}"
                    f", train_loss: {loss.item():.4f}"
                    f", step time: {(time.time() - step_start):.4f}"
                )
                
        for param_group in optimiser.param_groups:            
            print('lr=',param_group['lr'])
        # print('lr_scheduler.get_last_lr() = ',lr_scheduler.get_last_lr())
        if lr_cycling:
            lr_scheduler.step()
        
        epoch_loss /= step
        # epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        train_dice_scores.append(epoch_loss)
        
        
        if best_loss>epoch_loss:
            best_loss = epoch_loss
            state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimiser.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            }
            print(f'saving best loss model at epoch{epoch+1}')
            save_name=date.today().isoformat()+model_name+'_j'+str(job_id)+'_ts'+str(temporal_split)+ '_LL'
            saved_model=os.path.join(save_dir, save_name)
            torch.save(
                state,
                saved_model,
            )
                
            
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():                
                if training_mode=='val_exp_ens':
                    
                    sheet_name='Cluster_0'
                    save_name,best_metric[0],best_metric_epoch[0],metric[0]=validate(val_loader0,epoch,best_metric[0],best_metric_epoch[0],sheet_name)
                    if save_name not in model_names:
                        model_names.add(save_name)
                        best_metrics.add(best_metric[0])
                    sheet_name='Cluster_1'
                    save_name,best_metric[1],best_metric_epoch[1],metric[1]=validate(val_loader1,epoch,best_metric[1],best_metric_epoch[1],sheet_name)
                    if save_name not in model_names:
                        model_names.add(save_name)
                        best_metrics.add(best_metric[1])
                    sheet_name='Cluster_2'
                    save_name,best_metric[2],best_metric_epoch[2],metric[2]=validate(val_loader2,epoch,best_metric[2],best_metric_epoch[2],sheet_name)
                    if save_name not in model_names:
                        model_names.add(save_name)
                        best_metrics.add(best_metric[2])
                    sheet_name='Cluster_3'
                    save_name,best_metric[3],best_metric_epoch[3],metric[3]=validate(val_loader3,epoch,best_metric[3],best_metric_epoch[3],sheet_name)
                    if save_name not in model_names:
                        model_names.add(save_name)
                        best_metrics.add(best_metric[3])
                else:
                                      
                    save_name,best_metric,best_metric_epoch,metric=validate(val_loader,epoch,best_metric,best_metric_epoch,sheet_name)
                    val_scores.append(metric)
                    if best_met_old != best_metric:
                        best_met_old = best_metric
                        if training_mode == 'Infusion':
                            print('Added more noise to the mask')
                            train_loader.dataset.set_epoch(epoch) # directly setting the epoch in dataset class
                            
                    
                    if save_name not in model_names:
                        model_names.add(save_name)
                        best_metrics.add(best_metric)
                        
            
        wandb.log({'dice': metric, 'loss': epoch_loss})      
                 
        print(f"time consumption of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    
    del model
    # del optimiser
    gc.collect()
    torch.cuda.empty_cache()
    total_time = time.time() - total_start
    
    plt.figure()
    plt.plot(np.arange(total_epochs), train_dice_scores, 'k-', label='Train Dice Score')
    plt.plot(np.arange(total_epochs), val_scores, 'g-', label='Validation Dice Score')
    plt.title('Training and Validation Dice Scores')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.savefig('./lossDiceAndValDice.jpg')
    
    print(f"train completed, best_metric: {best_metric} at epoch: {best_metric_epoch}, total time: {total_time}")
    with open ('./time_consumption.csv', 'a') as sample:
        sample.write(f"{model_name},mode_{training_mode},{total_time},{date.today().isoformat()},{fold_num},{training_mode},{seed},{total_epochs}\n")
   
    return once

indices_dict={}
indices_dict['Train indices']=sorted(train_indices)
indices_dict['Test indices']=sorted(test_indices)
indices_dict['Val indices']=sorted(val_indices)
print(indices_dict)

if __name__=="__main__":
    if training_mode=='fs_ensemble':
        train_dataset = partition_dataset(data=EnsembleDataset('/scratch/a.bip5/BraTS 2021/selected_files_seed3.csv', transform=train_transform), shuffle=False, num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
        val_dataset = partition_dataset(data=EnsembleDataset('/scratch/a.bip5/BraTS 2021/selected_files_seed4.csv', transform=train_transform), num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
        trainingfunc(train_dataset, val_dataset)
        
    elif training_mode=='CV_fold':     
        
        full_dataset_train = BratsDataset(root_dir, transform=train_transform)
        full_dataset_val = BratsDataset(root_dir, transform=val_transform)
        # print(" cross val data set, CV_flag=1") # this is printed   
        train_dataset =Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices)
        trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir)
    elif training_mode=='atlas':     
        
        full_dataset_train = AtlasDataset(root_dir, transform = train_transform_isles)
        full_dataset_val = AtlasDataset(root_dir, transform=val_transform_isles)
        # print(" cross val data set, CV_flag=1") # this is printed   
        train_dataset =Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices)
        trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir)
    elif training_mode=='isles':  
        full_train=IslesDataset(root_dir ,transform= train_transform_isles )
        train_dataset = Subset(full_train, train_indices)        
        full_val = IslesDataset(root_dir ,transform=val_transform_isles )
        
        val_dataset = Subset(full_val, val_indices)
        trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir)
        
    elif training_mode=='ClusterBlend': 
       
        full_dataset_train = ClusterAugment(root_dir, transform=train_transform_CA)
        full_dataset_val = ClusterAugment(root_dir, transform=val_transform)
        # print(" cross val data set, CV_flag=1") # this is printed   
        train_dataset =Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices)
        trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir)
        
    elif training_mode=='Infusion':     
        class CustomSubset(Subset):
            def __init__(self, dataset, indices):
                super().__init__(dataset, indices)

            def set_epoch(self, epoch):
                # Directly call set_epoch on the original dataset
                self.dataset.set_epoch(epoch)
        full_dataset_train = BratsInfusionDataset(root_dir, transform=train_transform_infuse)
        full_dataset_val = BratsDataset(root_dir, transform=val_transform) #BratsDataset(root_dir, transform=val_transform)
        # print(" cross val data set, CV_flag=1") # this is printed   
        train_dataset = CustomSubset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices)#Subset(full_dataset_val, val_indices,)
        trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir)
  
        
    elif training_mode=='CustomActivation': 
        full_dataset_train = BratsDataset(root_dir, transform=train_transform)
        full_dataset_val = BratsDataset(root_dir, transform=val_transform)
        # print(" cross val data set, CV_flag=1") # this is printed   
        train_dataset =Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices,)
        trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir)
    elif training_mode=='SegResNetAtt':
        full_dataset_train = BratsDataset(root_dir, transform=train_transform)
        full_dataset_val = BratsDataset(root_dir, transform=val_transform)
        # print(" cross val data set, CV_flag=1") # this is printed   
        train_dataset =Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices,)
        trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir)

    elif training_mode=='PseudoAtlas':    
        
        full_dataset_train = BratsDataset(root_dir, transform=train_transform_PA)
        full_dataset_val = BratsDataset(root_dir, transform=val_transform_PA)
        print(" cross val data set Pseudo Atlas Training phase") # this is printed   
        train_dataset =Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices,)
        trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir)
    elif training_mode=='Flipper':    
        
        full_dataset_train = BratsDataset(root_dir, transform=train_transform_Flipper)
        full_dataset_val = BratsDataset(root_dir, transform=val_transform_Flipper)
        print(" cross val data set Pseudo Atlas Training phase") # this is printed   
        train_dataset =Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices,)
        trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir)
        # train_dataset = partition_dataset(data=train_dataset, transform=train_transform), shuffle=False, num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
        # val_dataset = partition_dataset(data=val_dataset, transform=train_transform), num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
      
        
    elif training_mode=='exp_ensemble':    
        print('Expert Ensemble going ahead now')
        # val_count=20
        train_count=100   
        
        xls = pd.ExcelFile(cluster_files)
         
        # Get all sheet names
        sheet_names = xls.sheet_names
        sheet_names=[x for x in sheet_names if 'Cluster_' in x]
        for sheet in sheet_names: 
            val_sheet_i=[]
            train_sheet_i=[]
            cluster_indices=pd.read_excel(cluster_files,sheet)['original index']
            sub_ids=pd.read_excel(cluster_files,sheet)['Index'].map(lambda x:x[-9:])
            map_dict=dict(zip(sub_ids,cluster_indices))
            
            # print(cluster_indices)
            for i,orig_i in enumerate(sorted(cluster_indices)):
                if orig_i in test_indices:
                   pass
                elif orig_i in val_indices:
                    # print('orig_i ',orig_i)
                    val_sheet_i.append(i)
                elif orig_i in train_indices:
                    # print(orig_i, 'training index')
                    train_sheet_i.append(i)
                    # print(train_sheet_i)
                    # print(len(train_sheet_i))
                else:
                    continue
            # print('train indices', train_sheet_i)
            # if len(val_sheet_i) < val_count:
                # raise(ValueError,f'Please reduce the number of val file count! Cluster {sheet} only had {len(val_sheet_i)} files')
            full_dataset_train = ExpDataset(cluster_files,sheet, transform=train_transform)
            full_dataset_val=ExpDataset(cluster_files,sheet, transform=val_transform)
            print('full dataset size',len(full_dataset_train))
            if fix_samp_num_exp:
                train_dataset =Subset(full_dataset_train, train_sheet_i[:exp_train_count])
                if super_val:
                    super_i=train_sheet_i[:exp_train_count]+val_sheet_i
                    val_dataset = Subset(full_dataset_val, super_i) 
                else:
                    val_dataset = Subset(full_dataset_val, val_sheet_i[:exp_val_count]) 
                # val_dataset = Subset(full_dataset, val_sheet_i)
                # val_dataset = Subset(full_dataset, val_sheet_i[:val_count])  
            else:
                train_dataset =Subset(full_dataset_train, train_sheet_i)           
                if super_val:
                    super_i=train_sheet_i+val_sheet_i
                    val_dataset = Subset(full_dataset_val, super_i) 
                else:
                    val_dataset = Subset(full_dataset_val, val_sheet_i) 
            print(len(train_dataset),len(val_dataset),'train and val')
            print('train_sheet i',len(train_sheet_i))
            trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir,sheet_name=sheet,map_dict=map_dict)
            gc.collect()
            
    elif training_mode=='PixelLayer':     
        
        full_dataset_train = BratsDataset(root_dir, transform=train_transform)
        full_dataset_val = BratsDataset(root_dir, transform=val_transform)
        # print(" cross val data set, CV_flag=1") # this is printed   
        train_dataset =Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices)
        trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir)
        
    elif training_mode=='val_exp_ens':
        full_dataset_train = BratsDataset(root_dir, transform=train_transform)
        train_dataset =Subset(full_dataset_train, train_indices)
        print('Expert by validation going ahead now')
          
        xls = pd.ExcelFile(cluster_files)
        
         
        # Get all sheet names
        sheet_names = xls.sheet_names
        sheet_names=[x for x in sheet_names if 'Cluster_' in x]
        val_sets=[]
        for sheet in sheet_names: 
            val_sheet_i=[]
            train_sheet_i=[]
            cluster_indices=pd.read_excel(cluster_files,sheet)['original index']
            sub_ids=pd.read_excel(cluster_files,sheet)['Index'].map(lambda x:x[-9:])
            map_dict=dict(zip(sub_ids,cluster_indices))
            
            for i,orig_i in enumerate(sorted(cluster_indices)):
                if orig_i in test_indices:
                   pass
                elif orig_i in val_indices:
                    # print('orig_i ',orig_i)
                    val_sheet_i.append(i)
                elif orig_i in train_indices:
                    # print(orig_i, 'training index')
                    pass
                else:
                    continue
            full_dataset_val=ExpDataset(cluster_files,sheet, transform=val_transform)
            val_dataset = Subset(full_dataset_val, val_sheet_i) 
            
            val_sets.append(val_dataset)
        trainingfunc_simple(train_dataset, val_sets,save_dir=save_dir,sheet_name=sheet)
        
    elif training_mode=='LayerNet':
        full_dataset_train = BratsDataset(root_dir, transform=train_transform)
        full_dataset_val = BratsDataset(root_dir, transform=val_transform)
         
        train_dataset =Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices,)
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=workers)
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=workers)
        model=model.to(device) 
        summary(model,(4,192,192,144))
        for epoch in range(total_epochs):
            epoch_start = time.time()
            model.train()
            print(f"epoch {epoch + 1}/{total_epochs}")
            epoch_loss = 0
            step = 0
            for ix ,batch_data in enumerate(train_loader):          
                # print(ix, 'just printing to see what up')
                if epoch==0:
                    if 'map_dict' in kwargs:
                        map_dict=kwargs['map_dict']
                        print(batch_data['id'])
                        for sid in batch_data['id']:
                            if training_mode=='exp_ensemble':
                                sample_index=map_dict[sid]
                                assert sample_index in train_indices , 'Training outside index'
                    
                step_start = time.time()
                step += 1
                inputs, masks = (
                    batch_data["image"].to(device),
                    batch_data["mask"].to(device),
                )
                optimiser.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs, loss_array = model(inputs)
                    loss=sum(loss_array)
    elif training_mode=='LoadNet':
        full_dataset_train = BratsDataset(root_dir, transform=train_transform)
        full_dataset_val = BratsDataset(root_dir, transform=val_transform)
         
        train_dataset =Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices,)
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=workers)
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=workers)
        model=model.to(device) 
        new_state_dict=torch.load(load_path,map_location=lambda storage,loc:storage.cuda(0))
        filtered_state_dict = {key: value for key, value in new_state_dict.items() if 'norm' not in key and 'up' not in key and 'conv_final' not in key}
        current_state_dict=model.state_dict()
        
        for key, value in new_state_dict.items():       
            if key in current_state_dict:
                current_state_dict[key] = value
                
        model.load_state_dict(current_state_dict,strict=False)
        optimiser =torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
        summary(model,(4,192,192,144))
        
        trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir)

    elif training_mode == 'intersect': 
        print('TRAINING INTERSECT')
        from Training.train_functions import trainingfunc_intersect
        full_dataset_train = BratsDataset(root_dir, transform=train_transform)
        full_dataset_val = BratsDataset(root_dir, transform=val_transform)
        # print(" cross val data set, CV_flag=1") # this is printed   
        train_dataset =Subset(full_dataset_train, train_indices)
     
        trainingfunc_intersect(train_dataset, save_dir=save_dir)

     
            
        
    else:
        print(' Choose a training method first in the config file!')

    torch.cuda.empty_cache()
    gc.collect()
    #storing everything to a csv row
    log_run_details(config_dict,model_names,best_metrics)

    #print the script at the end of every run
    script_path = os.path.abspath(__file__) # Gets the absolute path of the current script
    with open(script_path, 'r') as script_file:
               script_content = script_file.read()
    print("\n\n------ Script Content ------\n")
    print(script_content)
    print("\n---------------------------\n")

            
