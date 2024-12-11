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
isles_list,
train_transform_BP,
factor_increment
)
import torch.distributed as dist
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
training_samples,
output_path,
data_list_file_path,
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
from monai.data import Dataset
from Input.dataset import (
BratsDataset,
BratsDatasetPretrain,
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
from Evaluation.visualisation_functions import plot_zero
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai import transforms
import multiprocessing as mp
from monai.auto3dseg.utils import datafold_read
import logging


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
# Get a dictionary of the current global namespace
config_path = current_file.parents[1] / 'Input' / 'config.py'
with open(config_path, 'r') as config_file:
        script_content = config_file.read()
namespace = {}
exec(script_content, {}, namespace)

config_dict = dict()
job_id = os.environ.get('SLURM_JOB_ID', 'N/A')
config_dict['job_id']=job_id
for name, value in sorted(namespace.items()):
    if not name.startswith("__"):
        if type(value) in [str,int,float,bool]:
            print(f"{name}: {value}")
            config_dict[f"{name}"]=value
wandb.init(   
    project="segmentation",  # set the wandb project where this run will be logged
    name = job_id,
    notes = os.environ.get('NOTE_FOR_WANDB', 'N/A'),    
    config=config_dict # track hyperparameters and run metadata
)

current_file = Path(__file__).resolve() # This gets the root path two directories up


os.environ['PYTHONHASHSEED']=str(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
set_determinism(seed=seed)
device = torch.device(f"cuda:0")


unfreeze_layers=training_layers[unfreeze:]
model_names=set() #to store unique model_names saved by the script
best_metrics=set()
once=0



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



lr_scheduler =WarmupCosineSchedule(
                optimizer=optimiser, warmup_steps=3, warmup_multiplier=0.1, t_total=total_epochs
            ) #torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max) #torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max) 
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
                
    
   
    else: 
        try:
            
            model,optimiser,scaler,lr_scheduler,start_epoch = model_loader(load_path,train=True,scaler=scaler,lr_scheduler=lr_scheduler)           
            print('LOADED STATE SUCCESSFULLY')
            
        except:
            model = model_loader(load_path,train=True)
            print('LOADED STATE UNSUCCESSFULLY')
            optimiser = get_optimiser(model)
            scaler = torch.cuda.amp.GradScaler()
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max) 
        print("loaded saved model ", load_path)
        if PRUNE_PERCENTAGE is not None:
            model = prune_network(model)
            print('PRUNED MODEL!?!')
        if train_partial==True:
            model = partial_training(unfreeze_layers,model)
else:
    if training_mode=='ClusterBlend':
        model = ClusterBlend(model).to(device)
    elif training_mode=='PixelLayer':
        model = PixelLayer(model).to(device)
        
mp.set_start_method("fork", force=True)  # lambda functions fail to pickle without it
distributed = dist.is_initialized()

if torch.cuda.is_available():
    device = torch.device(rank)
    if distributed and dist.get_backend() == dist.Backend.NCCL:
        torch.cuda.set_device(rank)
else:
    device = torch.device("cpu")

def get_shared_memory_list(length=0):
    ## from auto3dseg generated codebase
    mp.current_process().authkey = np.arange(32, dtype=np.uint8).tobytes()
    shl0 = mp.Manager().list([None] * length)

    if distributed:
        # to support multi-node training, we need check for a local process group
        is_multinode = False

        if dist_launched():
            local_world_size = int(os.getenv("LOCAL_WORLD_SIZE"))
            world_size = int(os.getenv("WORLD_SIZE"))
            group_rank = int(os.getenv("GROUP_RANK"))
            if world_size > local_world_size:
                is_multinode = True
                # we're in multi-node, get local world sizes
                lw = torch.tensor(local_world_size, dtype=torch.int, device=device)
                lw_sizes = [torch.zeros_like(lw) for _ in range(world_size)]
                dist.all_gather(tensor_list=lw_sizes, tensor=lw)

                src = g_rank = 0
                while src < world_size:
                    # create sub-groups local to a node, to share memory only within a node
                    # and broadcast shared list within a node
                    group = dist.new_group(ranks=list(range(src, src + local_world_size)))
                    if group_rank == g_rank:
                        shl_list = [shl0]
                        dist.broadcast_object_list(shl_list, src=src, group=group, device=device)
                        shl = shl_list[0]
                    dist.destroy_process_group(group)
                    src = src + lw_sizes[src].item()  # rank of first process in the next node
                    g_rank += 1

        if not is_multinode:
            shl_list = [shl0]
            dist.broadcast_object_list(shl_list, src=0, device= device)
            shl = shl_list[0]

    else:
        shl = shl0

    return shl
    
def checkpoint_save(ckpt, model, **kwargs):
    save_time = time.time()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    config = config_with_relpath()

    torch.save({"state_dict": state_dict, "config": config, **kwargs}, ckpt)

    save_time = time.time() - save_time
    print(f"Saving checkpoint process: {ckpt}, {kwargs}, save_time {save_time:.2f}s")

    return save_time
    
def checkpoint_load(ckpt, model, **kwargs):
    if not os.path.isfile(ckpt):
        if global_rank == 0:
            warnings.warn("Invalid checkpoint file: " + str(ckpt))
    else:
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        epoch = checkpoint.get("epoch", 0)
        best_metric = checkpoint.get("best_metric", 0)

       
 
        print(
            f"=> loaded checkpoint {ckpt} (epoch {epoch}) (best_metric {best_metric}) setting start_epoch {epoch]}"
        )

def get_avail_cpu_memory():
    avail_memory = psutil.virtual_memory().available

    # check if in docker
    memory_limit_filename = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    if os.path.exists(memory_limit_filename):
        with open(memory_limit_filename, "r") as f:
            docker_limit = int(f.read())
            avail_memory = min(docker_limit, avail_memory)  # could be lower limit in docker

    return avail_memory  

def get_train_loader( data, cache_rate=0, persistent_workers=False):
    
    num_workers = num_workers
    batch_size = batch_size

    train_transform = train_transform_isles

    if cache_rate > 0:
        runtime_cache = get_shared_memory_list(length=len(data))
        train_ds = CacheDataset(
            data=data,
            transform=train_transform,
            copy_cache=False,
            cache_rate=cache_rate,
            runtime_cache=runtime_cache,
        )
    else:
        train_ds = Dataset(data=data, transform=train_transform)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
        persistent_workers=persistent_workers and num_workers > 0,
        pin_memory=True,
    )

    return train_loader

def get_val_loader( data, cache_rate=0, resample_label=False, persistent_workers=False):
    distributed = dist.is_initialized()
    num_workers = workers
    
    val_transform = val_transform_isles

    if cache_rate > 0:
        runtime_cache = get_shared_memory_list(length=len(data))
        val_ds = CacheDataset(
            data=data, transform=val_transform, copy_cache=False, cache_rate=cache_rate, runtime_cache=runtime_cache
        )
    else:
        val_ds = Dataset(data=data, transform=val_transform)

    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
        persistent_workers=persistent_workers and num_workers > 0,
        pin_memory=True,
    )

    return val_loader
    
def run_segmenter_worker(rank=0):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    dist_available = dist.is_available()
    global_rank = rank

    if dist_available:
        mgpu = override.get("mgpu", None)
        if mgpu is not None:
            logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
            dist.init_process_group(backend="nccl", rank=rank, timeout=timedelta(seconds=5400), **mgpu)
            mgpu.update({"rank": rank, "global_rank": rank})
            if rank == 0:
                print(f"Distributed: initializing multi-gpu tcp:// process group {mgpu}")

        elif dist_launched() and torch.cuda.device_count() > 1:
            rank = int(os.getenv("LOCAL_RANK"))
            global_rank = int(os.getenv("RANK"))
            world_size = int(os.getenv("LOCAL_WORLD_SIZE"))
            logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
            dist.init_process_group(backend="nccl", init_method="env://")  # torchrun spawned it
            override["mgpu"] = {"world_size": world_size, "rank": rank, "global_rank": global_rank}

            print(f"Distributed launched: initializing multi-gpu env:// process group {override['mgpu']}")

    segmenter = Segmenter(config_file=config_file, config_dict=override, rank=rank, global_rank=global_rank)
    best_metric = segmenter.run()
    segmenter = None

    if dist_available and dist.is_initialized():
        dist.destroy_process_group()

    return best_metric
    
def validate(val_loader, epoch, best_metric, best_metric_epoch, sheet_name=None, save_name=None, custom_inference=None):
    def default_inference(val_data):
        val_inputs = val_data["image"].to(device)
        val_masks = val_data["mask"].to(device)
        
        val_data["pred"] = inference(val_inputs, model)
        # print(val_data['pred'][0].shape,"val_data['pred'][0].shape)")
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
            #need to invert image as well for plotting purposes
            inverter = transforms.Invertd(keys="image", transform=val_transform_isles, orig_keys="image", meta_keys="image_meta_dict", nearest_interp=False, to_tensor=True)
            
            val_inputs = [inverter(x)["image"] for x in decollate_batch(val_data)]
            
            output_dir = os.path.join(output_path, job_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # plot_zero(val_inputs,val_outputs,val_masks,output_dir,job_id,'001')
            
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
        
        save_name_sd=date.today().isoformat()+model_name+'_j'+str(job_id)+'_ts'+str(temporal_split)+ '_sd'
        saved_model_sd=os.path.join(save_dir, save_name_sd)
        torch.save(
            model.state_dict(),
            saved_model_sd,
        )
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
    
    ####Datalist OVERRIDE
    if data_list_file_path: 
        data_file_base_dir= '/scratch/a.bip5/BraTS/dataset-ISLES22^public^unzipped^version/auto3d_scripts/data'
        training_files, validation_files= datafold_read(
                datalist=data_list_file_path, basedir=data_file_base_dir, fold=0
                )
        class ListDataset(Dataset):
            def __init__(self,data_list,transform=None):            
                self.data_list=data_list 
                self.transform=transform  
            def __len__(self):
                return min(max_samples,len(self.data_list))#        
            def __getitem__(self,idx):              
                item_dict=self.data_list[idx] 
                item_dict['mask']=item_dict.pop('label')
                if self.transform:                    
                    item_dict['id'] = item_dict['mask'].split('/')[-1]
                    item_dict=self.transform(item_dict) 
                return item_dict    
            
        train_dataset= ListDataset(training_files ,transform= train_transform_isles )
        val_dataset = ListDataset(validation_files ,transform=val_transform_isles )
 
    
    
    last_model=None
    print("number of files processed: ", train_dataset.__len__()) #this is not
    
  
   
    train_loader = get_train_loader(train_dataset, cache_rate= 1, persistant_workers=True )
    print('loading val data')
    if training_mode=='val_exp_ens':
        val_loader0=DataLoader(val_dataset[0], batch_size=batch_size, shuffle=False,num_workers=workers)
        val_loader1=DataLoader(val_dataset[1], batch_size=batch_size, shuffle=False,num_workers=workers)
        val_loader2=DataLoader(val_dataset[2], batch_size=batch_size, shuffle=False,num_workers=workers)
        val_loader3=DataLoader(val_dataset[3], batch_size=batch_size, shuffle=False,num_workers=workers)
    else:
        val_loader = get_val_loader(val_dataset, cache_rate=1,resample_label=True, persistent_workers=True)
    print("All Datasets assigned")                                             

    

        
    
    


    # use amp to accelerate training

    
    best_metric = -1
    best_loss = 2
    init_loss=2
    best_loss_epoch = -1
    best_metric_epoch = -1
    # best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []
    metric=-1
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
                transform_list= isles_list
                if epoch==0:
                    
                    updated_transform_isles = update_transforms_for_epoch(transform_list,init_loss=2,best_loss=2,patience=1)
                    full_train=IslesDataset("/scratch/a.bip5/BraTS/dataset-ISLES22^public^unzipped^version"  ,transform= updated_transform_isles )
                    train_dataset = Subset(full_train, train_indices)   # okay since train indices=230 on load_save
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=workers ) 
                    
                if load_save==1:
                    if epoch == start_epoch:
                        best_loss=1
                        print('AUGMENTATION UPDATE')
                        
                        updated_transform_isles = update_transforms_for_epoch(transform_list,init_loss,best_loss,patience=1)

                        full_train=IslesDataset("/scratch/a.bip5/BraTS/dataset-ISLES22^public^unzipped^version"  ,transform= updated_transform_isles )
                        train_dataset = Subset(full_train, train_indices)   # okay since train indices=230 on load_save
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=workers ) 
                if epoch==best_loss_epoch:                   
                    print('init_losss, best_loss', init_loss, best_loss)
                    updated_transform_isles = update_transforms_for_epoch(transform_list,init_loss,best_loss,patience=1)
                   
                    new_indices=indexes[:new_samples]
                    
                    print('AUGMENTATION UPDATE')
                    
                    full_train=IslesDataset("/scratch/a.bip5/BraTS/dataset-ISLES22^public^unzipped^version"  ,transform= updated_transform_isles )
                    if load_save==1:
                        train_dataset = Subset(full_train, train_indices)
                    else:
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
                    
                
        indices = list(range(len(train_indices)))#range(len(train_dataset))
        np.random.shuffle(indices)
        val_indices = list(range(50))#indexes[len(train_indices):]
        np.random.shuffle(val_indices)
        if use_sampler:
            # Define the size of the subset you want to use each epoch
            subset_size = 200  # Adjust this to whatever size you want
            subset_size_val = 50  # Adjust this to whatever size you want
            # Create the sampler
            sampler = SubsetRandomSampler(indices[:subset_size])
           
            train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=workers, sampler=sampler)
            
            sampler_val = SubsetRandomSampler(val_indices[:subset_size_val])
            
            val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=workers, sampler=sampler_val)
                
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
                            mask = F.interpolate(masks[bnum,:,:,:,:].unsqueeze(0), size = output.shape[-3:], mode='nearest')
                            masks_resized.append(mask)
                        mask_resized = torch.cat(masks_resized,dim=0)
                        # output_softmax= torch.softmax(output.float(),dim=1)
                        # print('mask_resized.shape, output_softmax.shape',mask_resized.shape, output_softmax.shape)
                        loss = loss_function(output,mask_resized)
                        losses.append(loss * weights[i])
                    loss = sum(losses)
                                       
                else:
                    loss = loss_function(outputs, masks) 
                    # print(type(outputs))
                    # print(loss, 'original loss')
                    edgy_loss_calc = edgy_dice_loss(outputs, masks)
                    # print(edgy_loss_calc, 'calculated edgy dice loss')
                          
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
        train_dice_scores.append(epoch_loss)
        
        
        if best_loss>epoch_loss:
            if epoch==0:
                if load_save==0:
                    init_loss = epoch_loss                
            print('lowest loss so far')
            best_loss = epoch_loss
            best_loss_epoch = epoch+1
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
            save_name_sd=date.today().isoformat()+model_name+'_j'+str(job_id)+'_ts'+str(temporal_split)+ '_LL_sd'
            saved_model_sd=os.path.join(save_dir, save_name_sd)
            torch.save(
                model.state_dict(),
                saved_model_sd,
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
        
    elif training_mode=='pretrain':  
        full_train=BratsDatasetPretrain(root_dir ,transform= train_transform_BP )
        train_dataset = Subset(full_train, train_indices)        
        full_val = IslesDataset(root_dir ,transform=val_transform_isles )
        
        val_dataset = Subset(full_val, val_indices)
        trainingfunc_simple(full_train, val_dataset,save_dir=save_dir)

        
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

            
