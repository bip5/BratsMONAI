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
import torch
from Input.config import load_path,VAL_AMP,roi,DE_option,root_dir
from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine
from monai.data import DataLoader,decollate_batch
import pandas as pd
import copy
from datetime import datetime
import cv2
from Input.dataset import make_dataset
from Evaluation.eval_functions import inference,dice_metric_ind,dice_metric_ind_batch,dice_metric,dice_metric_batch,model_loader, model_selector,load_cluster_centres,model_in_list,eval_single_raw
from monai.data import Dataset
from Input.dataset import train_indices,val_indices,test_indices



## At this point we have two dictionaries new and old with each key representing the index and value being the tuple with image and masks. This is different to the previous pipeline where the images and masks were separate lists. 



def check_membership(index, train,val, test):
    if index in train:
        return 'train'
    elif index in test:
        return 'test'
    elif index in val:
        return 'val'
    
dice_metric_new = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind_new = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind_batch_new = DiceMetric(include_background=True, reduction="mean_batch")
dice_metric_batch_new = DiceMetric(include_background=True, reduction="mean_batch") 

dice_metric_newold = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind_newold = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind_batch_newold = DiceMetric(include_background=True, reduction="mean_batch")
dice_metric_batch_newold = DiceMetric(include_background=True, reduction="mean_batch")   

dice_metric_oldnew = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind_oldnew = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind_batch_oldnew = DiceMetric(include_background=True, reduction="mean_batch")
dice_metric_batch_oldnew = DiceMetric(include_background=True, reduction="mean_batch")  

dice_metric_GTGT = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind_GTGT = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind_batch_GTGT = DiceMetric(include_background=True, reduction="mean_batch")
dice_metric_batch_GTGT = DiceMetric(include_background=True, reduction="mean_batch") 
 
 
dice_metric_ind_delta = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind_batch_delta = DiceMetric(include_background=True, reduction="mean_batch")

def evaluate_time_samples(load_path,old_loader,new_loader,modelweight_folder_path,expert=True,ensemble=False):
    model=model_loader(load_path)
    device = torch.device("cuda:0")        
    model.to(device)
    model.eval()
    ind_scores = dict()
    ind_scores_new = dict()
    ind_scores_newold = dict()
    ind_scores_oldnew = dict()
    ind_scores_GTGT = dict()
    
    if expert:
        base_model = model_loader(load_path)
        base_model.eval()
        cluster_centres,min_bound,max_bound = load_cluster_centres()
    dist_lists_old = {}
    dist_lists_new = {}
    with torch.no_grad():
        for i,(old_data,new_data) in enumerate (zip(old_loader,new_loader)): # each image
           
            if expert:
                model_list = model_in_list(modelweight_folder_path)
                if ensemble:
                    old_data['pred'] = torch.zeros_like(old_data['mask']).to(device)
                    new_data['pred'] = torch.zeros_like(new_data['mask']).to(device)
                    for model in model_list:
                        old_data["pred"] += eval_single_raw(model,old_data)
                        new_data["pred"] += eval_single_raw(model,new_data)
                    old_data["pred"] = old_data["pred"]/len(model_list)
                    new_data["pred"] = new_data["pred"]/len(model_list)
                    sub_index=old_data["index"]
                    sub_id = old_data["id"][0]
                    old_data=[post_trans(ii) for ii in decollate_batch(old_data)]
                    old_outputs,old_labels = from_engine(["pred","mask"])(old_data)
                    new_data=[post_trans(ii) for ii in decollate_batch(new_data)]
                    new_outputs,new_labels=from_engine(["pred","mask"])(new_data)
                else:
                    current_dice_old, tc_old,wt_old,et_old, old_outputs, old_labels,model_index,dist_lists_old= model_selector(model_list,old_data,base_model,min_bound,max_bound,cluster_centres,dist_lists_old)
                    current_dice_new, tc_new,wt_new,et_new, new_outputs, new_labels,model_index,dist_lists_new= model_selector(model_list,new_data,base_model,min_bound,max_bound,cluster_centres,dist_lists_new)
                    sub_index=old_data["index"]
                    sub_id = old_data["id"][0]
                
              
            else:
                old_inputs = old_data["image"].to(device) # pass to gpu
                sub_id = old_data["id"][0]
                sub_index=old_data["index"]
                

                old_data["pred"] = inference(old_inputs,model)
                old_data=[post_trans(ii) for ii in decollate_batch(old_data)] #returns a list of n tensors where n=batch_size
                old_outputs,old_labels = from_engine(["pred","mask"])(old_data) # returns two lists of tensors
                new_inputs = new_data["image"].to(device) # pass to gpu
                sub_id_new = new_data["id"][0]
                sub_index_new=new_data["index"]
                

                new_data["pred"] = inference(new_inputs,model)
                new_data=[post_trans(ii) for ii in decollate_batch(new_data)] #returns a list of n tensors where n=batch_size
                
                old_outputs,old_labels = from_engine(["pred","mask"])(old_data) # returns two lists of 
                new_outputs,new_labels=from_engine(["pred","mask"])(new_data)
                            
            for idx, y in enumerate(old_labels):
                old_labels[idx] = (y > 0.5).int()
            for idx, y in enumerate(old_outputs):
                old_outputs[idx] = (y > 0.5).int()
            for idx, y in enumerate(new_labels):
                new_labels[idx] = (y > 0.5).int()    
            for idx, y in enumerate(new_outputs):
                new_outputs[idx] = (y > 0.5).int()
                    
            old_outputs = [tensor.to(device) for tensor in old_outputs]
            old_labels = [tensor.to(device) for tensor in old_labels]
            new_outputs = [tensor.to(device) for tensor in new_outputs]
            new_labels = [tensor.to(device) for tensor in new_labels]
            # old_labels=old_data["mask"].to(device)
        
            old_volume_gt = [tensor.sum() for tensor in old_labels][0].item() #Batch Size=1
            new_volume_gt = [tensor.sum() for tensor in new_labels][0].item() #Batch size=1
            gt_delta = new_volume_gt - old_volume_gt
            
            old_volume_pred = [tensor.sum() for tensor in old_outputs][0].item()
            new_volume_pred = [tensor.sum() for tensor in new_outputs][0].item()
            pred_delta  = new_volume_pred - old_volume_pred
            
            delta_delta = pred_delta - gt_delta
            # torch.cuda.empty_cache()
            # print(type(old_outputs),old_labels[0].shape)
            
            ### ^ = bitwise XOR : ie how much of the change are we actually segmenting
            change_GT = [tensor1^tensor2 for tensor1,tensor2 in zip(old_labels,new_labels)]
            change_pred = [tensor1^tensor2 for tensor1,tensor2 in zip(old_outputs,new_outputs)]
            
            #OLD sample OLD TARGET
            dice_metric_ind(y_pred=old_outputs, y=old_labels)
            dice_metric_ind_batch(y_pred=old_outputs, y=old_labels)
            
            dice_metric(y_pred=old_outputs, y=old_labels)
            dice_metric_batch(y_pred=old_outputs, y=old_labels)
            
            dice_metric_ind_delta(y_pred=change_pred, y=change_GT)
            dice_metric_ind_batch_delta(y_pred=change_pred, y=change_GT)
            
            dice_metric_ind_GTGT(y_pred=old_labels, y=new_labels)
            dice_metric_ind_batch_GTGT(y_pred=old_labels, y=new_labels)
            
            dice_metric_GTGT(y_pred=old_labels, y=new_labels)
            dice_metric_batch_GTGT(y_pred=old_labels, y=new_labels)
            
            # NEW SAMPLE NEW TARGET
            dice_metric_ind_new(y_pred=new_outputs, y=new_labels)
            dice_metric_ind_batch_new(y_pred=new_outputs, y=new_labels)
            
            dice_metric_new(y_pred=new_outputs, y=new_labels)
            dice_metric_batch_new(y_pred=new_outputs, y=new_labels)
            
            # OLD SAMPLE NEW TARGET
            dice_metric_ind_oldnew(y_pred=old_outputs, y=new_labels)
            dice_metric_ind_batch_oldnew(y_pred=old_outputs, y=new_labels)
            
            dice_metric_oldnew(y_pred=old_outputs, y=new_labels)
            dice_metric_batch_oldnew(y_pred=old_outputs, y=new_labels)
            
            #NEW SAMPLE OLD TARGET
            dice_metric_ind_newold(y_pred=new_outputs, y=old_labels)
            dice_metric_ind_batch_newold(y_pred=new_outputs, y=old_labels)
            
            dice_metric_newold(y_pred=new_outputs, y=old_labels)
            dice_metric_batch_newold(y_pred=new_outputs, y=old_labels)
            
            membership_old=check_membership(sub_index.item(),train=train_indices,val=val_indices,test=test_indices)
            membership_new=check_membership(sub_index.item()+1,train=train_indices,val=val_indices,test=test_indices)
            marker_colors=['green','blue','red','orange']
            
            if membership_old == membership_new:
                if membership_old == 'train':
                    color = 'green'
                else: 
                    color = 'blue'
            else:   
                if membership_old == 'train':
                    color = 'red'
                else:
                    color = 'orange'
            
            
            # Compute the Dice score for this test case
            # current_dice = dice_metric_ind.aggregate().item()
            current_dice = dice_metric_ind.aggregate(reduction=None).item()
            batch_ind = dice_metric_ind_batch.aggregate()
            tc,wt,et = batch_ind[0].item(),batch_ind[1].item(),batch_ind[2].item()
            
            current_dice_delta = dice_metric_ind_delta.aggregate(reduction=None).item()
            batch_ind_delta = dice_metric_ind_batch_delta.aggregate()
            tc_delta,wt_delta,et_delta = batch_ind_delta[0].item(),batch_ind_delta[1].item(),batch_ind_delta[2].item()
            ind_scores[sub_id] = {'average':round(current_dice,4), 'tc':round(tc,4),'wt':round(wt,4),'et':round(et,4),'index': sub_index.item(), 'old in': membership_old, 'new in': membership_new,'d average':round(current_dice_delta,4), 'd tc':round(tc_delta,4),'d wt':round(wt_delta,4),'d et':round(et_delta,4), 'predicted volume delta': delta_delta, 'old volume gt': old_volume_gt, 'new volume gt': new_volume_gt, 'GT delta': gt_delta, 'old volume pred': old_volume_pred,'new volume pred': new_volume_pred, 'pred delta':pred_delta, 'marker_color': color}
            
            current_dice_GTGT = dice_metric_ind_GTGT.aggregate(reduction=None).item()
            batch_ind_GTGT = dice_metric_ind_batch_GTGT.aggregate()
            tc_GTGT,wt_GTGT,et_GTGT = batch_ind_GTGT[0].item(),batch_ind_GTGT[1].item(),batch_ind_GTGT[2].item()
            ind_scores_GTGT[sub_id] = {'average':round(current_dice_GTGT,4), 'tc':round(tc_GTGT,4),'wt':round(wt_GTGT,4),'et':round(et_GTGT,4),'index': sub_index.item()}
            
            current_dice_new = dice_metric_ind_new.aggregate(reduction=None).item()
            batch_ind_new = dice_metric_ind_batch_new.aggregate()
            tc_new,wt_new,et_new = batch_ind_new[0].item(),batch_ind_new[1].item(),batch_ind_new[2].item()
            ind_scores_new[sub_id] = {'average':round(current_dice_new,4), 'tc':round(tc_new,4),'wt':round(wt_new,4),'et':round(et_new,4),'index': sub_index.item()}
            
            current_dice_oldnew = dice_metric_ind_oldnew.aggregate(reduction=None).item()
            batch_ind_oldnew = dice_metric_ind_batch_oldnew.aggregate()
            tc_oldnew,wt_oldnew,et_oldnew = batch_ind_oldnew[0].item(),batch_ind_oldnew[1].item(),batch_ind_oldnew[2].item()
            ind_scores_oldnew[sub_id] = {'average':round(current_dice_oldnew,4), 'tc':round(tc_oldnew,4),'wt':round(wt_oldnew,4),'et':round(et_oldnew,4),'index': sub_index.item()}
            
            current_dice_newold = dice_metric_ind_newold.aggregate(reduction=None).item()
            batch_ind_newold = dice_metric_ind_batch_newold.aggregate()
            tc_newold,wt_newold,et_newold = batch_ind_newold[0].item(),batch_ind_newold[1].item(),batch_ind_newold[2].item()
            ind_scores_newold[sub_id] = {'average':round(current_dice_newold,4), 'tc':round(tc_newold,4),'wt':round(wt_newold,4),'et':round(et_newold,4),'index': sub_index.item()}
            
            
            
            dice_metric_ind.reset()
            dice_metric_ind_batch.reset()
            dice_metric_ind_delta.reset()
            dice_metric_ind_batch_delta.reset()
            dice_metric_ind_GTGT.reset()
            dice_metric_ind_batch_GTGT.reset()
            dice_metric_ind_new.reset()
            dice_metric_ind_batch_new.reset()
            dice_metric_ind_batch_newold.reset()
            dice_metric_ind_batch_newold.reset()
            dice_metric_ind_oldnew.reset()
            dice_metric_ind_batch_oldnew.reset()
                
            
        
        metric_org = dice_metric.aggregate().item()
        
        metric_batch_org = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()
        
        metric_org_GTGT = dice_metric_GTGT.aggregate().item()
        
        metric_batch_org_GTGT = dice_metric_batch_GTGT.aggregate()

        dice_metric_GTGT.reset()
        dice_metric_batch_GTGT.reset()
        
        metric_org_new = dice_metric_new.aggregate().item()
        
        metric_batch_org_new = dice_metric_batch_new.aggregate()

        dice_metric_new.reset()
        dice_metric_batch_new.reset()
        
        metric_org_oldnew = dice_metric_oldnew.aggregate().item()
        
        metric_batch_org_oldnew = dice_metric_batch_oldnew.aggregate()

        dice_metric_oldnew.reset()
        dice_metric_batch_oldnew.reset()
        
        metric_org_newold = dice_metric_newold.aggregate().item()
        
        metric_batch_org_newold = dice_metric_batch_newold.aggregate()

        dice_metric_newold.reset()
        dice_metric_batch_newold.reset()
        
        
        metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()
        
        metric_tc_GTGT, metric_wt_GTGT, metric_et_GTGT = metric_batch_org_GTGT[0].item(), metric_batch_org_GTGT[1].item(), metric_batch_org_GTGT[2].item()
        
        metric_tc_new, metric_wt_new, metric_et_new = metric_batch_org_new[0].item(), metric_batch_org_new[1].item(), metric_batch_org_new[2].item()
        
        metric_tc_newold, metric_wt_newold, metric_et_newold = metric_batch_org_newold[0].item(), metric_batch_org_newold[1].item(), metric_batch_org_newold[2].item()
        
        metric_tc_oldnew, metric_wt_oldnew, metric_et_oldnew = metric_batch_org_oldnew[0].item(), metric_batch_org_oldnew[1].item(), metric_batch_org_oldnew[2].item()
        
        print("Metric on original image spacing old old : ", metric_org)
        print(f"metric_tc: {metric_tc:.4f}", f"   metric_wt: {metric_wt:.4f}", f"   metric_et: {metric_et:.4f}")
        
        print("Metric on original image spacing new new : ", metric_org_new)
        print(f"metric_tc: {metric_tc_new:.4f}", f"   metric_wt: {metric_wt_new:.4f}", f"   metric_et: {metric_et_new:.4f}")
        print("Metric on original image spacing new new : ", metric_org_newold)
        print(f"metric_tc: {metric_tc_newold:.4f}", f"   metric_wt: {metric_wt_newold:.4f}", f"   metric_et: {metric_et_newold:.4f}")
        print("Metric on original image spacing new new : ", metric_org_oldnew)
        print(f"metric_tc: {metric_tc_oldnew:.4f}", f"   metric_wt: {metric_wt_oldnew:.4f}", f"   metric_et: {metric_et_oldnew:.4f}")
        
        return ind_scores,ind_scores_new,ind_scores_newold,ind_scores_oldnew,ind_scores_GTGT

        
        