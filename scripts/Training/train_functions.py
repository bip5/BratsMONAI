import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/') # need to add the import folder to system path for python to know where to look for

import numpy as np
import os
import torch
from monai.data import DataLoader
from Input.localtransforms import (
train_transform,
train_transform_infuse,
train_transform_PA,
train_transform_Flipper,
val_transform,
val_transform_PA,
val_transform_Flipper,
post_trans,
)

from prun import prune_network
from monai.handlers.utils import from_engine
from Input.config import (

root_dir,
weights_dir,
total_epochs,
val_interval,
VAL_AMP,
fold_num,
seed,
batch_size,
model_name,
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
no_val
)

from Training.loss_function import loss_function,edgy_dice_loss
from Training.network import model,create_model 
from Training.running_log import log_run_details
from Evaluation.eval_functions import model_loader
from Evaluation.evaluation import (
inference,

)
from Evaluation.evaluation import (
dice_metric,
dice_metric_batch,
inference,
)

from Input.dataset import (
BratsDataset,
BratsInfusionDataset,
EnsembleDataset,
ExpDataset,
train_indices,
val_indices,
test_indices
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

''' To be used to store training functions for various experiments'''

device = torch.device(f"cuda")

# Define your function to save checkpoints
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

# Define your function to load checkpoints
def load_checkpoint(filename="checkpoint.pth.tar"):
    return torch.load(filename)
    
# Function to calculate mean variance of model layers
def calculate_mean_variance(state_dict):
    variances = []
    for param in state_dict.values():
        variances.append(torch.var(param).item())
    return np.mean(variances)

def trainingfunc_intersect(train_dataset, save_dir, model=model, sheet_name=None, once=None, **kwargs):
    last_model = None
    print("number of files processed: ", len(train_dataset))
  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    print("All Datasets assigned")

    np.random.seed(seed+21)
    torch.cuda.manual_seed(seed+21)
    set_determinism(seed=seed+21)    
    torch.manual_seed(seed+21)
    model = create_model().to(device)
    model = torch.nn.DataParallel(model)

    with torch.cuda.amp.autocast():
        summary(model, (4, 192, 192, 128))

    # Load state_dict for variance calculation
    state_dict = load_checkpoint(load_path)
    pretrained_variance = calculate_mean_variance(state_dict)
    
    optimiser = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max)
    
    best_loss = float('inf')
    best_variance_delta = float('inf')
    epoch_loss_values = []
 
    print("starting epochs")
    
    patience_counter = 0
    saved_models_count = 0
    last_saved_models_count = 0
    model_performance_dict = {}
    best_met_old = 0
    train_dice_scores = []
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{total_epochs}")

        model.train()
        epoch_loss = 0
        step = 0                                     
        
        for ix, batch_data in enumerate(train_loader):
            step_start = time.time()
            step += 1
            inputs, masks = (
                batch_data["image"].to(device),
                batch_data["mask"].to(device),
            )
            optimiser.zero_grad()
                
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, masks)
                          
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            
            epoch_loss += loss.item()
            if step % 10 == 0:
                print(
                    f"{step}/{len(train_dataset) // train_loader.batch_size}"
                    f", train_loss: {loss.item():.4f}"
                    f", step time: {(time.time() - step_start):.4f}"
                )
                
        for param_group in optimiser.param_groups:            
            print('lr=', param_group['lr'])
        
        if lr_cycling:
            lr_scheduler.step()
        
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        train_dice_scores.append(epoch_loss)
        
        # Calculate mean variance of the current model
        current_variance = calculate_mean_variance(model.state_dict())
        variance_delta = abs(pretrained_variance - current_variance)

        if best_loss > epoch_loss: 
            if best_variance_delta > variance_delta:
                best_loss = epoch_loss
                best_variance_delta = variance_delta
                print(f'saving best loss model at epoch {epoch + 1} with variance delta of {best_variance_delta}')
                save_name = f"{date.today().isoformat()}_{model_name}_j{job_id}_ts{temporal_split}_nv_is_e{epoch + 1}"
                saved_model = os.path.join(save_dir, save_name)
                torch.save(model.state_dict(), saved_model)
                
        print(f"time consumption of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    total_time = time.time() - total_start
    
    with open('./time_consumption.csv', 'a') as sample:
        sample.write(f"{model_name},mode_{training_mode},{total_time},{date.today().isoformat()},{fold_num},{training_mode},{seed},{total_epochs}\n")
   
    return once

def trainingfunc_intersect_2model(train_dataset, save_dir, sheet_name=None, once=None, **kwargs):
    last_model = None
    print("number of files processed: ", len(train_dataset))
  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
   
    print("All Datasets assigned")                                             

    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    set_determinism(seed=seed)    
    torch.manual_seed(seed)
    model = create_model().to(device)
    model = torch.nn.DataParallel(model)

    np.random.seed(seed + 21)
    torch.cuda.manual_seed(seed + 21)
    set_determinism(seed=(seed + 21))    
    torch.manual_seed(seed + 21)
    model2 = create_model().to(device)
    model2 = torch.nn.DataParallel(model2)

    with torch.cuda.amp.autocast():
        summary(model, (4, 192, 192, 128))

    if load_save == 1:        
        model = model_loader(load_path)

    optimiser = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max)
    
    optimiser2 = torch.optim.Adam(model2.parameters(), lr, weight_decay=1e-5)
    scaler2 = torch.cuda.amp.GradScaler()
    lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser2, T_0=T_max)
    
    best_metric = -1
    best_loss = 1
    best_loss2 = 1
    best_metric_epoch = -1
    epoch_loss_values = []
    best_weightsum_delta = float('inf')
 
    print("starting epochs")
    
    patience_counter = 0
    saved_models_count = 0
    last_saved_models_count = 0
    model_performance_dict = {}
    best_met_old = 0
    train_dice_scores = []
    train_dice_scores2 = []
    val_scores = []
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{total_epochs}")

        # Train model 1
        model.train()
        epoch_loss = 0
        step = 0                                     
        
        for ix, batch_data in enumerate(train_loader):
            step_start = time.time()
            step += 1
            inputs, masks = (
                batch_data["image"].to(device),
                batch_data["mask"].to(device),
            )
            optimiser.zero_grad()
                
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, masks)
                          
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            
            epoch_loss += loss.item()
            if step % 10 == 0:
                print(
                    f"{step}/{len(train_dataset) // train_loader.batch_size}"
                    f", train_loss: {loss.item():.4f}"
                    f", step time: {(time.time() - step_start):.4f}"
                )
                
        for param_group in optimiser.param_groups:            
            print('lr=', param_group['lr'])
        
        if lr_cycling:
            lr_scheduler.step()
        
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        train_dice_scores.append(epoch_loss)
        
        # Save model 1 checkpoint
        checkpoint_path = os.path.join(save_dir, f"model1_epoch_{epoch + 1}.pth.tar")
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimiser.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
        }, checkpoint_path)

        # Clear GPU memory
        del model, optimiser, scaler, lr_scheduler
        gc.collect()
        torch.cuda.empty_cache()

        # Load model 2 checkpoint
        model2_checkpoint_path = os.path.join(save_dir, f"model2_epoch_{epoch}.pth.tar")
        if os.path.exists(model2_checkpoint_path):
            checkpoint = load_checkpoint(model2_checkpoint_path)
            model2.load_state_dict(checkpoint['state_dict'])
            optimiser2.load_state_dict(checkpoint['optimizer'])
            scaler2.load_state_dict(checkpoint['scaler'])
            lr_scheduler2.load_state_dict(checkpoint['scheduler'])

        # Train model 2
        model2.train()
        epoch_loss2 = 0
        step = 0                                     
        
        for ix, batch_data in enumerate(train_loader):
            step_start = time.time()
            step += 1
            inputs, masks = (
                batch_data["image"].to(device),
                batch_data["mask"].to(device),
            )
            optimiser2.zero_grad()
                
            with torch.cuda.amp.autocast():
                outputs2 = model2(inputs)
                loss2 = loss_function(outputs2, masks)
                          
            scaler2.scale(loss2).backward()
            scaler2.step(optimiser2)
            scaler2.update()
            
            epoch_loss2 += loss2.item()
            if step % 10 == 0:
                print(
                    f"{step}/{len(train_dataset) // train_loader.batch_size}"
                    f", train_loss: {loss2.item():.4f}"
                    f", step time: {(time.time() - step_start):.4f}"
                )
                
        for param_group in optimiser2.param_groups:            
            print('lr=', param_group['lr'])
        
        if lr_cycling:
            lr_scheduler2.step()
        
        epoch_loss2 /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss2:.4f}")
        train_dice_scores2.append(epoch_loss2)
        
        # Save model 2 checkpoint
        checkpoint_path = os.path.join(save_dir, f"model2_epoch_{epoch + 1}.pth.tar")
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model2.state_dict(),
            'optimizer': optimiser2.state_dict(),
            'scaler': scaler2.state_dict(),
            'scheduler': lr_scheduler2.state_dict(),
        }, checkpoint_path)

        # Clear GPU memory
        del model2, optimiser2, scaler2, lr_scheduler2
        gc.collect()
        torch.cuda.empty_cache()

        # Load model 1 checkpoint for the next epoch
        model1_checkpoint_path = os.path.join(save_dir, f"model1_epoch_{epoch + 1}.pth.tar")
        checkpoint = load_checkpoint(model1_checkpoint_path)
        model = create_model().to(device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])
        optimiser = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
        optimiser.load_state_dict(checkpoint['optimizer'])
        scaler = torch.cuda.amp.GradScaler()
        scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max)
        lr_scheduler.load_state_dict(checkpoint['scheduler'])

        # Load the saved models for comparison
        model_checkpoint_path = os.path.join(save_dir, f"model1_epoch_{epoch + 1}.pth.tar")
        model2_checkpoint_path = os.path.join(save_dir, f"model2_epoch_{epoch + 1}.pth.tar")
        checkpoint1 = load_checkpoint(model_checkpoint_path)
        checkpoint2 = load_checkpoint(model2_checkpoint_path)
        model1_weights = sum(torch.sum(param).item() for param in checkpoint1['state_dict'].values())
        model2_weights = sum(torch.sum(param).item() for param in checkpoint2['state_dict'].values())
        weight_sum_difference = abs(model1_weights - model2_weights)

        if no_val:
            if best_loss > epoch_loss or best_loss2 > epoch_loss2:
                best_loss = min(epoch_loss, epoch_loss2)
                if best_weightsum_delta > weight_sum_difference:
                    best_weightsum_delta = weight_sum_difference                   
                    print(f'saving best loss model at epoch {epoch + 1} with weightsum delta of {best_weightsum_delta}')
                    save_name = f"{date.today().isoformat()}_{model_name}_j{job_id}_ts{temporal_split}_nv_is_e{epoch + 1}"
                    saved_model = os.path.join(save_dir, save_name)
                    torch.save(model.state_dict(), saved_model)
                
        print(f"time consumption of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    total_time = time.time() - total_start
    
    with open('./time_consumption.csv', 'a') as sample:
        sample.write(f"{model_name},mode_{training_mode},{total_time},{date.today().isoformat()},{fold_num},{training_mode},{seed},{total_epochs}\n")
   
    return once
    
    
def trainingfunc_intersectSingle(train_dataset, save_dir, sheet_name=None, once=None, **kwargs):
    last_model = None
    print("number of files processed: ", len(train_dataset))
  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    
   
    print("All Datasets assigned")                                             

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    set_determinism(seed=seed)    
    torch.manual_seed(seed)
    model = create_model().to(device)
    model = torch.nn.DataParallel(model)

    np.random.seed(seed + 21)
    torch.cuda.manual_seed(seed + 21)
    set_determinism(seed=(seed + 21))    
    torch.manual_seed(seed + 21)
    model2 = create_model().to(device)
    model2 = torch.nn.DataParallel(model2)

    with torch.cuda.amp.autocast():
        summary(model, (4, 192, 192, 128))

    if load_save == 1:        
        model = model_loader(load_path)

    optimiser = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    print("Model defined and passed to GPU")
    torch.backends.cudnn.benchmark = True

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max)
    
    optimiser2 = torch.optim.Adam(model2.parameters(), lr, weight_decay=1e-5)
    scaler2 = torch.cuda.amp.GradScaler()
    lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser2, T_0=T_max)
    
    best_metric = -1
    best_loss = 1
    best_loss2 = 1
    best_metric_epoch = -1
    epoch_loss_values = []
    best_weightsum_delta = float('inf')
 
    print("starting epochs")
    
    patience_counter = 0
    saved_models_count = 0
    last_saved_models_count = 0
    model_performance_dict = {}
    best_met_old = 0
    train_dice_scores = []
    train_dice_scores2 = []
    val_scores = []
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{total_epochs}")
        model.train()
        model2.train()
        epoch_loss = 0
        epoch_loss2 = 0
        step = 0                                     
        
        for ix, batch_data in enumerate(train_loader):
            step_start = time.time()
            step += 1
            inputs, masks = (
                batch_data["image"].to(device),
                batch_data["mask"].to(device),
            )
            optimiser.zero_grad()
            optimiser2.zero_grad()
                
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                outputs2 = model2(inputs)
                
                loss = loss_function(outputs, masks)
                loss2 = loss_function(outputs2, masks)
                          
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            
            scaler2.scale(loss2).backward()
            scaler2.step(optimiser2)
            scaler2.update()
            
            epoch_loss += loss.item()
            epoch_loss2 += loss2.item()
            if step % 10 == 0:
                print(
                    f"{step}/{len(train_dataset) // train_loader.batch_size}"
                    f", train_loss: {loss.item():.4f}"
                    f", step time: {(time.time() - step_start):.4f}"
                )
                
        for param_group in optimiser.param_groups:            
            print('lr=', param_group['lr'])
        
        if lr_cycling:
            lr_scheduler.step()
            lr_scheduler2.step()
        
        epoch_loss /= step
        epoch_loss2 /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        train_dice_scores.append(epoch_loss)
        train_dice_scores2.append(epoch_loss2)
      
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            weight_sum = sum(torch.sum(param).item() for param in model.state_dict().values())
            weight_sum2 = sum(torch.sum(param).item() for param in model2.state_dict().values())
            weight_sum_difference = abs(weight_sum - weight_sum2)
            if best_weightsum_delta > weight_sum_difference:
                best_weightsum_delta = weight_sum_difference                   
                print(f'saving best loss model at epoch {epoch + 1} with weightsum delta of {best_weightsum_delta}')
                save_name = f"{date.today().isoformat()}_{model_name}_j{job_id}_ts{temporal_split}_nv_is_e{epoch + 1}"
                saved_model = os.path.join(save_dir, save_name)
                torch.save(model.state_dict(), saved_model)
                
        print(f"time consumption of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    total_time = time.time() - total_start
    
    with open('./time_consumption.csv', 'a') as sample:
        sample.write(f"{model_name},mode_{training_mode},{total_time},{date.today().isoformat()},{fold_num},{training_mode},{seed},{total_epochs}\n")
   
    return once
