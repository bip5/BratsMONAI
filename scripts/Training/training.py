import sys
sys.path.append('/scratch/a.bip5/BraTS 2021/scripts/') # need to add the import folder to system path for python to know where to look for

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
freeze_patience
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
for name, value in namespace.items():
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

unfreeze_layers=training_layers[-unfreeze:]
model_names=set() #to store unique model_names saved by the script
best_metrics=set()
def check_gainless_epochs(epoch, best_metric_epoch):
    gainless_epochs = (epoch + 1) - best_metric_epoch
    if gainless_epochs >= freeze_patience:
        return True
    else:
        return False


def trainingfunc_simple(train_dataset, val_dataset,model=model,sheet_name=None):
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

    total_start = time.time()



   
    print("starting epochs")
    gainless_counter=0 #to check how many times gainless function has tripped
    for epoch in range(total_epochs):
        # train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{total_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
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
            # def closure():
                # optimiser.zero_grad()
                # with torch.cuda.amp.autocast():
                    # outputs = model(inputs)
                    # loss = loss_function(outputs, masks)
                # scaler.scale(loss).backward()
                # return loss                
            # optimiser.step(closure)
            # scaler.step(optimiser)
            # scaler.update()
                
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
                
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    
                    if CV_flag==1:                        
                        saved_model=os.path.join(root_dir, model_name+str(total_start)+"CV"+str(fold_num)+"ms"+str(max_samples)+"rs"+str(seed)+method+'ep'+str(epoch))
                        torch.save(
                            model.state_dict(),
                            saved_model
                        )
                    
                    elif sheet_name is not None:  
                       
                        save_name=date.today().isoformat()+'T'+str(total_start)+sheet_name+model_name
                        saved_model=os.path.join(root_dir, save_name)
                        torch.save(
                            model.state_dict(),
                           saved_model,
                            )
                        print(f'A NEW MASTER Is BORN named "{save_name}" ')
                            
                            
                    else:
                        print('NO CV or expert training sheet name might be none',sheet_name)
                        save_name=os.path.join(root_dir, date.today().isoformat()+'T'+str(total_start)+model_name)
                        torch.save(
                            model.state_dict(),
                            save_name,
                        )
                    
                    
                    print("saved new best metric model")
                    
                    
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
        if save_name not in model_names:
            model_names.add(save_name)
            best_metrics.add(best_metric)
        
        # if sheet_name:
            # if check_gainless_epochs(epoch, best_metric_epoch):
                # gainless_counter+=1
                # break
        if freeze_train:            
            if check_gainless_epochs(epoch, best_metric_epoch):
                gainless_counter+=1
                freeze_index=len(training_layers)%gainless_counter -1
                model.load_state_dict(torch.load(saved_model),strict=False)
                print(f'loaded {saved_model} to commence training')
                
                if isolate_layer:
                    freeze_layers=training_layers[freeze_index]
                    best_metric_epoch=epoch
                    
                    if len(freeze_layers)==len(training_layers):
                        print('Nothing left to freeze, this is probably as good as it gets, play with other hyps maybe?')
                        break
                    for param in model.parameters():
                        param.requires_grad = False
                    for name, param in model.named_parameters():
                        if name in freeze_layers:
                            param.requires_grad = True
                    print(f' only training layer {gainless_counter-1}, commencing training')
                else:
                    freeze_layers=training_layers[:gainless_counter]
                    best_metric_epoch=epoch
                    
                    if len(freeze_layers)==len(training_layers):
                        print('Nothing left to freeze, this is probably as good as it gets, play with other hyps maybe?')
                        break
                    # for param in model.parameters():
                        # param.requires_grad = False
                    
                    for name, param in model.named_parameters():
                        if name in freeze_layers:
                            param.requires_grad = False
                    print(f' froze {gainless_counter} items, commencing training')        
                

                
        print(f"time consumption of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    total_time = time.time() - total_start

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
    with open ('./time_consumption.csv', 'a') as sample:
        sample.write(f"{model_name},{method},{total_time},{date.today().isoformat()},{fold_num},{CV_flag},{seed},{total_epochs}\n")
   
    return None




    
def trainingfunc(train_dataset, val_dataset,model=model,sheet_name=None):
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()#int(os.getenv('WORLD_SIZE', '1'))
    print('total GPUs:',world_size)
    print("number of files processed: ", train_dataset.__len__()) #this is not
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=False,sampler=train_sampler,num_workers=workers)
    print('loading val data')
    # val_sampler = DistributedSampler(val_dataset,num_replicas=world_size, rank=rank) #/ not strictly required but makes things faster
    val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=workers)
    print("All Datasets assigned")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    model=model.to(device)  
    

    
    if sheet_name is None:
        with torch.cuda.amp.autocast():
            summary(model,(4,192,192,144))  
    torch.manual_seed(seed)    
    model=DistributedDataParallel(module=model, device_ids=[device],find_unused_parameters=False)
    if load_save==1:
        model.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(rank)),strict=True)
        print("loaded saved model ", load_path)
    optimiser =torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)#torch.optim.LBFGS(model.parameters()) 
    scaler = torch.cuda.amp.GradScaler()
    print("Model defined and passed to GPU")



    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max) #torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max) 

    # use amp to accelerate training

    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    total_start = round(time.time())



   
    print("starting epochs")
    for epoch in range(total_epochs):
        # train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{total_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
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
            # def closure():
                # optimiser.zero_grad()
                # with torch.cuda.amp.autocast():
                    # outputs = model(inputs)
                    # loss = loss_function(outputs, masks)
                # scaler.scale(loss).backward()
                # return loss                
            # optimiser.step(closure)
            # scaler.step(optimiser)
            # scaler.update()
                
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
        
        lr_scheduler.step()
        
        # epoch_loss /= step
        # epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        
        if (epoch + 1) % val_interval == 0 & rank==0:
            model.eval()
            with torch.no_grad():

                for val_data in val_loader:
                    val_inputs, val_masks = (
                        val_data["image"].to(device),
                        val_data["mask"].to(device),
                    )
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
                
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    if CV_flag==1:                        
                        saved_model=os.path.join(root_dir, model_name+str(total_start)+"CV"+str(fold_num)+"ms"+str(max_samples)+"rs"+str(seed)+method+'ep'+str(epoch))
                        torch.save(
                            model.state_dict(),
                            saved_model
                        )
                    
                    elif sheet_name is not None:  
                       
                        save_name=date.today().isoformat()+'T'+str(total_start)+sheet_name+model_name
                        torch.save(
                            model.state_dict(),
                            os.path.join(root_dir, save_name),
                            )
                        print(f'A NEW MASTER Is BORN named {save_name} ')
                            
                            
                    else:
                        print('NO CV or expert training sheet name might be none',sheet_name)
                        save_name=os.path.join(root_dir, date.today().isoformat()+'T'+str(total_start)+model_name)
                        torch.save(
                            model.state_dict(),
                            save_name,
                        )
                    
                    
                    print("saved new best metric model")
                    
                    
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
        
        gc.collect()
        torch.cuda.empty_cache()
        
                
                

                
        print(f"time consumption of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
    with open ('./time_consumption.csv', 'a') as sample:
        sample.write(f"{model_name},{method},{total_time},{date.today().isoformat()},{fold_num},{CV_flag},{seed},{total_epochs}\n")
   
    return None  


if fs_ensemble == 1:
    train_dataset = partition_dataset(data=EnsembleDataset('/scratch/a.bip5/BraTS 2021/selected_files_seed3.csv', transform=train_transform), shuffle=False, num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
    val_dataset = partition_dataset(data=EnsembleDataset('/scratch/a.bip5/BraTS 2021/selected_files_seed4.csv', transform=train_transform), num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
    trainingfunc(train_dataset, val_dataset)
elif CV_flag == 1:
    full_dataset_train = BratsDataset(root_dir, transform=train_transform)
    full_dataset_val = BratsDataset(root_dir, transform=val_transform)
    print(" cross val data set, CV_flag=1") # this is printed   
    train_dataset =Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_val, val_indices)
    trainingfunc_simple(train_dataset, val_dataset)
    # train_dataset = partition_dataset(data=train_dataset, transform=train_transform), shuffle=False, num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
    # val_dataset = partition_dataset(data=val_dataset, transform=train_transform), num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
    
elif exp_ensemble==1:
    print('Expert Ensemble going ahead now')
    val_count=20
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
        
        if len(val_sheet_i) < val_count:
            raise(f'Please reduce the number of val file count! Cluster {sheet} only had {len(val_sheet_i)} files')
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
        trainingfunc_simple(train_dataset, val_dataset,sheet_name=sheet)
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

        
