import sys
sys.path.append('/scratch/a.bip5/BraTS 2021/scripts/')

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
import torch
from Input.config import (
cluster_files,
VAL_AMP,
eval_path,
eval_mode,
batch_size,
workers,
test_samples_from,
eval_folder,
weights_dir,
output_path
)
from Input.dataset import ExpDataset,ExpDatasetEval,test_indices,train_indices,val_indices
from Input.localtransforms import test_transforms0,post_trans,train_transform,val_transform
from Training.running_log import log_run_details
import pandas as pd
from Training.network import model
from monai.data import DataLoader,decollate_batch
import numpy as np
from torch.utils.data import Subset
torch.multiprocessing.set_sharing_strategy('file_system')
import time
import os
import matplotlib.pyplot as plt
import datetime

namespace = locals().copy()
config_dict=dict()
for name, value in namespace.items():
    if type(value) in [str,int,float,bool]:
        print(f"{name}: {value}")
        config_dict[f"{name}"]=value

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

def inference(input,model):

    def _compute(input,model):
        return sliding_window_inference(
            inputs=input,
            roi_size=(192,192, 144),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input,model)
    else:
        return _compute(input,model)
        
def plot_prediction(image_slice,gt_mask,pred_mask,o_path,title):
    fig,ax = plt.subplots(1,2,figsize=(10,10))
    
    ax[0].imshow(image_slice, cmap='gray', origin= 'lower')
    ax[0].set_title('Original Slice')
    # mask_data=(pred_mask & gt_mask).int()
    
    # if np.any(mask_data) and not np.all(mask_data):  # Ensure there are both 1s and 0s
        # print(np.unique(mask_data))
        # ax.contour(mask_data, levels=[1])
    # else:
        # print("No contour can be drawn for this slice.")
        
    # sys.exit()
    ax[1].imshow(image_slice, cmap='gray', origin= 'lower')
    print(np.unique((pred_mask & gt_mask).astype(int)))
    # True positive (pred and groud truth overlap)
    ax[1].contour((pred_mask & gt_mask).astype(int), colors= 'g', levels=[1])
    print(np.unique((pred_mask & ~gt_mask).astype(int)))
    # False positives (predicted but not in ground truth)
    ax[1].contour((pred_mask & ~gt_mask).astype(int), colors='r' , levels=[1])
    print(np.unique((~pred_mask & gt_mask).astype(int)))
    # False negatives (present in ground truth but not predicted)
    ax[1].contour((~pred_mask & gt_mask).astype(int), colors='b', levels=[1])
    
    ax[1].set_title(title)  
    plt.savefig(os.path.join(o_path, title + '.png'))
    plt.close()
    
def find_centroid(mask_channel):
    """
    Compute the centroid slice indices along each view for a given mask channel.
    """
    coords = np.argwhere(mask_channel)
    return coords.mean(axis=0).astype(int)    
        

def evaluate(eval_path,test_loader,output_path,model=model):
    #
    device = torch.device("cuda:0")
    model=torch.nn.DataParallel(model)
    model.to(device)
    
    # model.load_state_dict(torch.load(eval_path),strict=False)
    model.load_state_dict(torch.load(eval_path, map_location=lambda storage, loc: storage.cuda(0)), strict=False)
    
    model.eval()

    with torch.no_grad():
    

        for test_data in test_loader: # each image
            test_inputs = test_data["image"].to(device) # pass to gpu
            test_labels=test_data["mask"].to(device)
            sub_id=test_data["id"]
            
            for idx, y in enumerate(test_labels):
                test_labels[idx] = (y > 0.5).int()
            
            test_outputs=inference(test_inputs,model)
            test_outputs=[post_trans(i) for i in decollate_batch(test_outputs)] 
                      
            dice_metric(y_pred=test_outputs, y=test_labels)
            dice_metric_batch(y_pred=test_outputs, y=test_labels)
            
            

             # Create a unique folder for each test case
            case_save_path = os.path.join(output_path, f"test_case_{sub_id[0]}")
            os.makedirs(case_save_path, exist_ok=True)
            
            for mask_channel in range(test_labels.shape[1]):  # loop over the 3 mask channels
                gt_channel = test_labels[0][mask_channel].cpu().numpy()
                pred_channel = test_outputs[0].cpu().numpy()[mask_channel]

                centroid = find_centroid(gt_channel)
                
                for view, axis in enumerate(["Axial", "Sagittal", "Coronal"]):
                    slice_idx = centroid[view]

                    if view == 0:  # Axial view
                        slice_data = test_data["image"][0][2, :, :, slice_idx].cpu().numpy()  # C=4, hence using channel 2
                        gt_slice = gt_channel[:, :, slice_idx] > 0.5
                        pred_slice = pred_channel[:, :, slice_idx] > 0.5
                    elif view == 1:  # Sagittal view
                        slice_data = test_data["image"][0][2, slice_idx, :, :].cpu().numpy()
                        gt_slice = gt_channel[slice_idx, :, :] > 0.5
                        pred_slice = pred_channel[slice_idx, :, :] > 0.5
                    else:  # Coronal view
                        slice_data = test_data["image"][0][2, :, slice_idx, :].cpu().numpy()
                        gt_slice = gt_channel[:, slice_idx, :] > 0.5
                        pred_slice = pred_channel[:, slice_idx, :] > 0.5

                    title = f"Channel_{mask_channel}_{axis}_Slice_{slice_idx}"
                    plot_prediction(slice_data, gt_slice, pred_slice, case_save_path, title)    
         
        metric_org = dice_metric.aggregate().item()
        
        metric_batch_org = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()

    metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()

    print("Metric on original image spacing: ", metric_org)
    print(f"metric_tc: {metric_tc:.4f}", f"   metric_wt: {metric_wt:.4f}", f"   metric_et: {metric_et:.4f}")
    return metric_org,metric_tc,metric_wt,metric_et

if __name__ =='__main__':
    if eval_mode=='cluster':
        test_count=30
        train_count=100
        
        xls = pd.ExcelFile(cluster_files) #to get sheets
        
       # Get all sheet names
        sheet_names = xls.sheet_names
        sheet_names=[x for x in sheet_names if 'Cluster' in x]
        resultdict=dict()
        model_names=[]
        
        for modelweights in os.listdir(eval_folder):
            eval_path = os.path.join(eval_folder,modelweights)
            model_names.append(modelweights)
            dice_scores=[]
            for sheet in sheet_names:
                test_sheet_i=[]
                train_sheet_i=[]
                val_sheet_i=[]
                cluster_indices=pd.read_excel(cluster_files,sheet)['original index']
                
                for i,orig_i in enumerate(cluster_indices):
                    if test_samples_from=='trainval':
                        if orig_i in train_indices:
                            # while len(train_sheet_i)<train_count:
                            train_sheet_i.append(i)
                        elif orig_i in val_indices:
                            val_sheet_i.append(i)
                    elif test_samples_from=='val':
                        if orig_i in val_indices:
                            val_sheet_i.append(i)
                        
                    else:                               
                        if orig_i in test_indices:
                           print('orig_i ',orig_i)
                           test_sheet_i.append(i)
                #will be adding empty list to give identical when mode not trainval
                test_sheet_i = test_sheet_i+train_sheet_i+val_sheet_i 
                
                evaluating_sheet=sheet
                print(f'Evaluating {len(test_sheet_i)} samples from {evaluating_sheet} with saved model{eval_path}')
                test_ds = ExpDataset(cluster_files,evaluating_sheet,transform=val_transform)#creating dataset from sheet in xls
                test_ds = Subset(test_ds,test_sheet_i)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)                
                
                # Get the current date and time
                now = datetime.datetime.now()

                # Convert it to a string in a specific format (e.g., YYYY-MM-DD_HH-MM-SS)
                formatted_time = sheet+now.strftime('%Y-%m-%d_%H-%M-%S')     

                # Create the new directory path using the formatted time
                new_dir = os.path.join(output_path, formatted_time)

                # Make the new directory
                os.makedirs(new_dir, exist_ok=True)
                metric_org,metric_tc,metric_wt,metric_et=evaluate(eval_path,test_loader,new_dir)
                
                dice_scores.append(metric_org)
                config_dict['metric_org'] = metric_org
                config_dict['metric_tc'] = metric_tc
                config_dict['metric_wt'] = metric_wt
                config_dict['metric_et'] = metric_et
                config_dict['Cluster'] = evaluating_sheet
                log_path='/scratch/a.bip5/BraTS 2021/eval_running_log.csv'
                log_run_details(config_dict,[],csv_file_path=log_path)
            resultdict['m'+ modelweights]=dice_scores
            
            
            
        folder_name=eval_folder.split('/')[-1]
              
        resultdict=pd.DataFrame(resultdict,index=sheet_names)
        resultdict.to_csv(f'./EvalCluster{folder_name}.csv')
    
    