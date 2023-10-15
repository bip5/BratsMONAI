import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/') # need to add the import folder to system path for python to know where to look for

import numpy as np
import os
import torch
from monai.data import DataLoader
from Input.localtransforms import (
train_transform,
val_transform,
post_trans,
)
from Input.config import (
CV_flag,
root_dir,
weights_dir,
total_epochs,
val_interval,
VAL_AMP,
method,
fold_num,
seed,
batch_size,
model_name,
T_max,
lr,
max_samples,
model_name,
fs_ensemble,
exp_ensemble,
workers,
load_save,
load_path,
DDP,
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
binsemble
)
from Evaluation.evaluation import (
inference,

)
from Training.loss_function import loss_function
from Training.network import model
from Training.running_log import log_run_details


from Evaluation.evaluation import (
dice_metric,
dice_metric_batch,
inference,
)

from Input.dataset import (
BratsDataset,
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
from torch.utils.data import Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
import gc

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
for name, value in sorted(namespace.items()):
    if type(value) in [str,int,float,bool]:
        print(f"{name}: {value}")
        config_dict[f"{name}"]=value
os.environ['PYTHONHASHSEED']=str(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
set_determinism(seed=seed)

training_layers=["module.convInit.conv.weight", "module.down_layers.0.1.conv1.conv.weight", "module.down_layers.0.1.conv2.conv.weight", "module.down_layers.1.0.conv.weight", "module.down_layers.1.1.conv1.conv.weight", "module.down_layers.1.1.conv2.conv.weight", "module.down_layers.1.2.conv1.conv.weight", "module.down_layers.1.2.conv2.conv.weight", "module.down_layers.2.0.conv.weight", "module.down_layers.2.1.conv1.conv.weight", "module.down_layers.2.1.conv2.conv.weight", "module.down_layers.2.2.conv1.conv.weight", "module.down_layers.2.2.conv2.conv.weight", "module.down_layers.3.0.conv.weight", "module.down_layers.3.1.conv1.conv.weight", "module.down_layers.3.1.conv2.conv.weight", "module.down_layers.3.2.conv1.conv.weight", "module.down_layers.3.2.conv2.conv.weight", "module.down_layers.3.3.conv1.conv.weight", "module.down_layers.3.3.conv2.conv.weight", "module.down_layers.3.4.conv1.conv.weight", "module.down_layers.3.4.conv2.conv.weight", "module.up_layers.0.0.conv1.conv.weight", "module.up_layers.0.0.conv2.conv.weight", "module.up_layers.1.0.conv1.conv.weight", "module.up_layers.1.0.conv2.conv.weight", "module.up_layers.2.0.conv1.conv.weight", "module.up_layers.2.0.conv2.conv.weight", "module.up_samples.0.0.conv.weight", "module.up_samples.0.1.deconv.weight", "module.up_samples.0.1.deconv.bias", "module.up_samples.1.0.conv.weight", "module.up_samples.1.1.deconv.weight", "module.up_samples.1.1.deconv.bias", "module.up_samples.2.0.conv.weight", "module.up_samples.2.1.deconv.weight", "module.up_samples.2.1.deconv.bias", "module.conv_final.2.conv.weight", "module.conv_final.2.conv.bias"]

unfreeze_layers=training_layers[unfreeze:]
model_names=set() #to store unique model_names saved by the script
best_metrics=set()
once=0
def check_gainless_epochs(epoch, best_metric_epoch):
    gainless_epochs = epoch  - best_metric_epoch
    if gainless_epochs >= freeze_patience:
        return True
    else:
        return False

now = datetime.now()
formatted_time =now.strftime('%Y-%m-%d_%H-%M-%S')

total_start = time.time()
save_dir=os.path.join(weights_dir,'m'+formatted_time)

os.makedirs(save_dir,exist_ok=True)
print('SAVING MODELS IN ', save_dir)

job_id = os.environ.get('SLURM_JOB_ID', 'N/A')

def trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir,model=model,sheet_name=None,once=once):
    last_model=None
    print("number of files processed: ", train_dataset.__len__()) #this is not
    
   
    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=workers)
    print('loading val data')
   
    val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=workers)
    print("All Datasets assigned")

    device = torch.device(f"cuda:0")
    # torch.cuda.set_device(device)
    model=model.to(device)  
    model=torch.nn.DataParallel(model)
    
    
    if sheet_name is None:
        with torch.cuda.amp.autocast():
            summary(model,(4,192,192,144))  
    else:    
        while once<1:
            with torch.cuda.amp.autocast():
                summary(model,(4,192,192,144))
            once+=1
        
    torch.manual_seed(seed)    
    # model=DistributedDataParallel(module=model, device_ids=[device],find_unused_parameters=False)
    if load_save==1:
        model.load_state_dict(torch.load(load_path),strict=False)
        print("loaded saved model ", load_path)
        if train_partial==True:
            # Step 2: Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
            
            for name, param in model.named_parameters():
                if name in unfreeze_layers:
                    param.requires_grad = True
            print(f'only training {unfreeze_layers}')
        
    
    optimiser =torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    print("Model defined and passed to GPU")
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True



    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max) #torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max) 

    # use amp to accelerate training

    
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    
    



   
    print("starting epochs")
    gainless_counter=0 #to check how many times gainless function has tripped, resets after all layers thawed at least once
    patience_counter=0 #separate counter which resets at checkpoint
    saved_models_count = 0
    last_saved_models_count=0
    model_performance_dict = {}  # to store {model_path: dice_score}
    
    for epoch in range(total_epochs):
        # train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{total_epochs}")
        model.train()
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
                        
        
        
        
        for batch_data in train_loader:
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
        
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():

                for val_data in val_loader:
                    val_inputs, val_masks = (
                        val_data["image"].to(device),
                        val_data["mask"].to(device),
                    )
                    for idx, y in enumerate(val_masks):
                        val_masks[idx] = (y > 0.5).int()
                    val_outputs = inference(val_inputs,model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_masks)
                    dice_metric_batch(y_pred=val_outputs, y=val_masks)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                metric_tc = metric_batch[0].item()
                metric_values_tc.append(metric_tc)
                metric_wt = metric_batch[1].item()
                metric_values_wt.append(metric_wt)
                metric_et = metric_batch[2].item()
                metric_values_et.append(metric_et)
                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    saved_models_count+=1
                
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    
                    if CV_flag==1:      
                        save_name=model_name+"CV"+str(fold_num)+'_j'+str(job_id)+'ep'+str(epoch)
                        saved_model=os.path.join(save_dir,save_name)
                        print(saved_model)
                        torch.save(
                            model.state_dict(),
                            saved_model
                        )
                    
                    elif sheet_name is not None:  
                       
                        save_name=sheet_name+'_'+str(load_save)+'_j'+str(job_id)
                        saved_model=os.path.join(save_dir, save_name)
                        torch.save(
                            model.state_dict(),
                           saved_model,
                            )
                        print(f'A NEW MASTER Is BORN named "{save_name}" ')
                            
                            
                    else:
                        print('NO CV or expert training sheet name might be none',sheet_name)
                        save_name=date.today().isoformat()+model_name+'_j'+str(job_id)
                        saved_model=os.path.join(save_dir, save_name)
                        torch.save(
                            model.state_dict(),
                            save_name,
                        )
                    
                    
                    print("saved new best metric model")
                    last_model,previous_best_model=saved_model,last_model
                    patience_counter=0
                    
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
            
            if binsemble:
                
                print('TRAINING BINSEMBLE \n \n \n \n \n')
                if patience_counter>freeze_patience:
                    if saved_models_count>last_saved_models_count:
                        if previous_best_model and last_model:
                            # Load weights from the two models
                            last_model_weights = torch.load(last_model)
                            previous_best_model_weights = torch.load(previous_best_model)
                            
                            # Average the weights
                            averaged_weights = {name: (last_model_weights[name] + previous_best_model_weights[name]) / 2 
                                                for name in last_model_weights}
                            
                            # Update the current model's weights
                            model.load_state_dict(averaged_weights)
                            print(f"Averaged weights of models {previous_best_model} and {last_model}.")
                            last_saved_models_count=saved_models_count
                    else:
                        print("Insufficient models to average. Consider saving initial models for averaging.")
                else:
                    patience_counter+=1
                
                
                
        # if sheet_name:
            # if check_gainless_epochs(epoch, best_metric_epoch):
                # gainless_counter+=1
                # break
        
                
                    
            

                
        print(f"time consumption of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    if save_name not in model_names:
        model_names.add(save_name)
        best_metrics.add(best_metric)
    del model
    del optimiser
    gc.collect()
    torch.cuda.empty_cache()
    total_time = time.time() - total_start

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
    with open ('./time_consumption.csv', 'a') as sample:
        sample.write(f"{model_name},{method},{total_time},{date.today().isoformat()},{fold_num},{CV_flag},{seed},{total_epochs}\n")
   
    return once




if fs_ensemble:
    train_dataset = partition_dataset(data=EnsembleDataset('/scratch/a.bip5/BraTS 2021/selected_files_seed3.csv', transform=train_transform), shuffle=False, num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
    val_dataset = partition_dataset(data=EnsembleDataset('/scratch/a.bip5/BraTS 2021/selected_files_seed4.csv', transform=train_transform), num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
    trainingfunc(train_dataset, val_dataset)
elif CV_flag:
    indices_dict={}
    indices_dict['Train indices']=train_indices
    indices_dict['Test indices']=test_indices
    indices_dict['Val indices']=val_indices
    print(indices_dict)
    
    full_dataset_train = BratsDataset(root_dir, transform=train_transform)
    full_dataset_val = BratsDataset(root_dir, transform=val_transform)
    print(" cross val data set, CV_flag=1") # this is printed   
    train_dataset =Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_val, val_indices,)
    trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir)
    # train_dataset = partition_dataset(data=train_dataset, transform=train_transform), shuffle=False, num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
    # val_dataset = partition_dataset(data=val_dataset, transform=train_transform), num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
    
elif exp_ensemble:
    print('Expert Ensemble going ahead now')
    # val_count=20
    train_count=100
    
    
    
    xls = pd.ExcelFile(cluster_files)
     
    # Get all sheet names
    sheet_names = xls.sheet_names
    sheet_names=[x for x in sheet_names if 'Cluster' in x]
    for sheet in sheet_names: 
        val_sheet_i=[]
        train_sheet_i=[]
        cluster_indices=pd.read_excel(cluster_files,sheet)['original index']
        for i,orig_i in enumerate(cluster_indices):
            if orig_i in test_indices:
               pass
            elif orig_i in val_indices:
                print('orig_i ',orig_i)
                val_sheet_i.append(i)
            else:
                train_sheet_i.append(i)
        
        # if len(val_sheet_i) < val_count:
            # raise(ValueError,f'Please reduce the number of val file count! Cluster {sheet} only had {len(val_sheet_i)} files')
        full_dataset_train = ExpDataset(cluster_files,sheet, transform=train_transform)
        full_dataset_val=ExpDataset(cluster_files,sheet, transform=val_transform)
        if fix_samp_num_exp:
            train_dataset =Subset(full_dataset_train, train_sheet_i[:exp_train_count])
            if super_val:
                super_i=train_sheet_i[:exp_train_count]+val_sheet_i
                val_dataset = Subset(full_dataset_val, super_i) 
            else:
                val_dataset = Subset(full_dataset_val, val_sheet_i) 
            # val_dataset = Subset(full_dataset, val_sheet_i)
            # val_dataset = Subset(full_dataset, val_sheet_i[:val_count])  
        else:
            train_dataset =Subset(full_dataset_train, train_sheet_i)
            if super_val:
                super_i=train_sheet_i+val_sheet_i
                val_dataset = Subset(full_dataset_val, super_i) 
            else:
                val_dataset = Subset(full_dataset_val, val_sheet_i) 
        trainingfunc_simple(train_dataset, val_dataset,save_dir=save_dir,sheet_name=sheet)
        gc.collect()
else:
    print(' Choose a training method first in the config file!')

#storing everything to a csv row
log_run_details(config_dict,model_names,best_metrics)

#print the script at the end of every run
script_path = os.path.abspath(__file__) # Gets the absolute path of the current script
with open(script_path, 'r') as script_file:
           script_content = script_file.read()
print("\n\n------ Script Content ------\n")
print(script_content)
print("\n---------------------------\n")

        
