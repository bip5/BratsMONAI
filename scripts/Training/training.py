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
model_name
)
from Evaluation.evaluation import (
inference,

)
from Training.loss_function import loss_function
from Training.network import model
from Training.optimiser import optimiser

from Evaluation.evaluation import (
dice_metric,
dice_metric_batch,
inference,
)

from Input.dataset import (
BratsDataset,
train_indices,
val_indices
)

from monai.utils import set_determinism
from monai.data import decollate_batch
from datetime import date, datetime
import time
from torchsummary import summary
from torch.utils.data import Subset
 

    
os.environ['PYTHONHASHSEED']=str(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
set_determinism(seed=seed)

    

train_dataset=BratsDataset("/scratch/a.bip5/BraTS 2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData"  ,transform=train_transform )
if CV_flag==1:
    print("loading cross val data")
    val_dataset=Subset(train_dataset,train_indices)
    train_dataset=Subset(train_dataset,train_indices)
    
else:     
    val_dataset=Subset(train_dataset,np.arange(10))
    train_dataset=Subset(train_dataset,[10])    
    
print("number of files processed: ", train_dataset.__len__())
train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=1)
val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=4)
print("All Datasets assigned")

device = torch.device("cuda:0")
torch.manual_seed(seed)
with torch.cuda.amp.autocast():
    summary(model,(4,192,192,144))
model=torch.nn.DataParallel(model)
optimiser = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()
print("Model defined and passed to GPU")



lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=T_max) 

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

total_start = time.time()

print("starting epochs")
for epoch in range(total_epochs):
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
            
    print('lr_scheduler.get_last_lr() = ',lr_scheduler.get_last_lr())
    if epoch>99:
        lr_scheduler.step()
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    
    if (epoch + 1) % val_interval == 0:
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
                    if epoch>99:
                        torch.save(
                            model.state_dict(),
                            os.path.join(root_dir, model_name+"CV"+str(fold_num)+"ms"+str(max_samples)+"rs"+str(seed)+method+'extra'))
                    else:
                    
                        torch.save(
                            model.state_dict(),
                            os.path.join(root_dir, model_name+"CV"+str(fold_num)+"ms"+str(max_samples)+"rs"+str(seed)+method)
                        )
                else:
                    torch.save(
                        model.state_dict(),
                        os.path.join(root_dir, date.today().isoformat()+'T'+str(datetime.today().hour)+ model_name),
                    )
                print("saved new best metric model")
                
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
            

            
    print(f"time consumption of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
with open ('./time_consumption.csv', 'a') as sample:
    sample.write(f"{model_name},{method},{total_time},{date.today().isoformat()},{fold_num},{CV_flag},{seed},{total_epochs}\n")
   
