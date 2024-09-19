import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')
import nibabel as nib


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
output_path,
slice_dice,
model_name,
plot_output,
load_path,
plot_single_slice,
eval_from_folder,
roi,
use_cluster_for_online_val,
root_dir,
plot_list,
plots_dir,
limit_samples,
base_perf_path
)
from Input.dataset import (
BratsDataset,
ExpDataset,ExpDatasetEval,
test_indices,train_indices,
val_indices,Brats23valDataset,BratsTimeDataset
)
from Input.localtransforms import test_transforms1,post_trans,train_transform,val_transform,post_trans_test,val_transform_Flipper
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
from Analysis.encoded_features import single_encode
from sklearn.preprocessing import MinMaxScaler
from monai.config import print_config
from monai.metrics import DiceMetric
from Evaluation.eval_functions import plot_scatter,plot_prediction,find_centroid,eval_model_selector
from Evaluation.eval_functions import (
inference,dice_metric_ind,dice_metric_ind_batch,
dice_metric,dice_metric_batch,plot_expert_performance,
distance_ensembler,model_loader, 
get_model_paths,
modelweight_sums
)
from Evaluation.eval_functions2 import evaluate_time_samples
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')
import os
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from monai.metrics import DiceMetric
from Input.localtransforms import post_trans
from Analysis.encoded_features import single_encode
from scipy.spatial import distance

from Input.config import load_path,VAL_AMP,roi,DE_option,dropout, TTA_ensemble,cluster_files, raw_features_filename
from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine
import copy
from datetime import datetime
import cv2
from Training.dropout import dropout_network
from monai.transforms import CenterSpatialCrop,SpatialPad,CropForeground,ScaleIntensity,AdjustContrast,Identity

import pytest

seed=0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
device = torch.device("cuda:0")
def test_get_model_paths():
    modelpaths = get_model_paths(eval_folder)
    assert len(modelpaths)==4 , f'{len(modelpaths)} in folder, expected 4'
    
def test_modelweight_sums():
    modelpaths = get_model_paths(eval_folder)
    weight_sums = modelweight_sums(modelpaths)
    
    assert len(set(weight_sums)) > 1 # should be true as long at at least two different values exist- will fail if all models are identical

def test_loaded_models():
    modelpaths = get_model_paths(eval_folder)
    model_list=[model_loader(x) for x in modelpaths]
    model_addresses = []
    for idx, model in enumerate(model_list):
        print(f"Model {idx} address: {id(model)}")
        model_addresses.append(id(model))
    assert len(model_addresses) == len(set(model_addresses)), 'model_addresses are identical'
    
dice_metric_ind = DiceMetric(include_background=True, reduction="mean") 
def test_consistency():
    '''Pass same model as separate models and check if dice score same each time with distance ensembler'''
    #folder below has 4 identical models with slight name differences
    eval_folder = '/scratch/a.bip5/BraTS/weights/job_7807898' 
    modelpaths = get_model_paths(eval_folder)
    model_list=[model_loader(x) for x in modelpaths]
    full_ds = BratsDataset(root_dir,transform = val_transform)
    test_ds = Subset(full_ds,test_indices[:1])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4) 
    dice_scores = []
    
    with torch.no_grad():
        for i,test_data in enumerate(test_loader):
            test_input = test_data["image"].to(device)
            for j in range(len(model_list)):
            
                try:
                    model_selected = model_list[j]          
                    model_selected.eval()
                except:
                    print(f'NOT possible to load model {j+1}')
                    sys.exit()
                    continue
                test_data["pred"] = inference(test_input,model_selected)
                decollated_raw= decollate_batch(test_data)
                post_data = post_trans(decollated_raw[0])
                
                output = [post_data["pred"].to(device)]
                label = [post_data["mask"].to(device)]
                dice_metric_ind(y_pred= output, y=label)
                current_dice = dice_metric_ind.aggregate(reduction=None).item()
                print(f'Dice score for model {j} is {current_dice}')
                dice_metric_ind.reset()
                dice_scores.append(current_dice)
    
    assert len(set(dice_scores))==1, 'same model generating multiple scores for same sample!'
    
    
    
    
    
    
    
    
    
    
    
    