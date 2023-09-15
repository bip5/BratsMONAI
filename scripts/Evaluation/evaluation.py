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
workers
)
from Input.dataset import ExpDatasetEval,test_indices
from Input.localtransforms import test_transforms0,post_trans
import pandas as pd
from Training.network import model
from monai.data import DataLoader,decollate_batch
import numpy as np
from torch.utils.data import Subset



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
        

def evaluate(eval_path,test_loader,model=model):
    # model.load_state_dict(torch.load(eval_path),strict=False)
    device = torch.device("cuda:0")
    model.to(device)
    model=torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(eval_path, map_location=lambda storage, loc: storage.cuda(0)), strict=True)
    
    model.eval()

    with torch.no_grad():

        for test_data in test_loader: # each image
            test_inputs = test_data["image"].to(device) # pass to gpu
            test_labels=test_data["label"].to(device)
            
            test_outputs=inference(test_inputs,model)
            test_outputs=[post_trans(i) for i in decollate_batch(test_outputs)] 
                      
            dice_metric(y_pred=test_outputs, y=test_labels)
            dice_metric_batch(y_pred=test_outputs, y=test_labels)            
        metric_org = dice_metric.aggregate().item()
        
        metric_batch_org = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()

    metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()

    print("Metric on original image spacing: ", metric_org)
    print(f"metric_tc: {metric_tc:.4f}", f"   metric_wt: {metric_wt:.4f}", f"   metric_et: {metric_et:.4f}")
    return None

if __name__ =='__main__':
    if eval_mode=='cluster':
        test_count=30
        test_sheet_i=[]
        xls = pd.ExcelFile(cluster_files) #to get sheets
        
       # Get all sheet names
        sheet_names = xls.sheet_names
        sheet_names=[x for x in sheet_names if 'Cluster' in x]
        
        
        for sheet in sheet_names:
            cluster_indices=pd.read_excel(cluster_files,sheet)['original index']
            for i,orig_i in enumerate(cluster_indices):
                if orig_i in test_indices:
                   print('orig_i ',orig_i)
                   test_sheet_i.append(i)
            evaluating_sheet=sheet
            print(f'Evaluating {test_count} samples from {evaluating_sheet} with saved model{eval_path}')
            test_ds=ExpDatasetEval(cluster_files,evaluating_sheet,transform=test_transforms0)#creating dataset from sheet in xls
            test_ds=Subset(test_ds,test_sheet_i[:test_count])
            test_loader= DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers) 
            evaluate(eval_path,test_loader)
    
    
    