import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')
import os
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from monai.metrics import DiceMetric
from Input.localtransforms import post_trans,post_trans_test
from Analysis.encoded_features import single_encode, single_encode_tfmr
from Analysis.encoded_features_x5 import single_encode_x5
from scipy.spatial import distance
import torch
from Input.config import encoder_path,load_path,VAL_AMP,roi,DE_option,dropout, TTA_ensemble,cluster_files, raw_features_filename,inf_overlap,model_name,output_path,jit_model
from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine
from monai.data import DataLoader,decollate_batch
import pandas as pd
import copy
from datetime import datetime
import cv2
from Training.dropout import dropout_network
from Training.optimiser import get_optimiser
from monai.transforms import CenterSpatialCrop,SpatialPad,CropForeground,ScaleIntensity,AdjustContrast,Identity
from BraTS2023Metrics.metrics import get_LesionWiseResults as lesion_wise
import nibabel as nib
from Input.dataset import (
BratsDataset,
ExpDataset,ExpDatasetEval,
test_indices,train_indices,
val_indices,Brats23valDataset,BratsTimeDataset
)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind_batch = DiceMetric(include_background=True, reduction="mean_batch")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
features_df=pd.read_csv('/scratch/a.bip5/BraTS/trainFeatures_01April.csv') # to get the max and min values to save for later
features_df=features_df.drop(columns=['Unnamed: 0','mask_path','subject_id'],errors='ignore') 
max_bound=np.load(f'/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers/{raw_features_filename}_max_bound.npy')
min_bound=np.load(f'/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers/{raw_features_filename}_min_bound.npy')

seed=0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
device = torch.device("cuda:0")
    
def inference(input,model):

    def _compute(input,model):
        
        return sliding_window_inference(
            inputs=input,
            mode="gaussian",
            roi_size=roi,
            sw_batch_size=1,
            predictor=model,
            overlap=inf_overlap,
            )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input,model)
    else:
        return _compute(input,model)
        

def model_loader(modelweight_path,model_id=model_name,train=False,lr_scheduler=None,scaler=None,start_epoch=None):

    from Training.network import create_model #should hopefully solve the issue
    # Create the model instance
    model = create_model(model_id)
    
    
    def wrap_model(state_dict,model):
        # Check if the state dict contains keys prefixed with 'module.'
        # This indicates that the model was saved with DataParallel
        is_dataparallel = any(key.startswith('module.') for key in state_dict.keys())

        if is_dataparallel:
            # Wrap the model with DataParallel before loading the state dict
            model = torch.nn.DataParallel(model)
            model.load_state_dict(state_dict, strict=True)
        else:
            # If there's no 'module.' prefix, load the state dict as is
            # This also handles the case where the model needs to be wrapped but the saved model wasn't
            # If necessary, you can modify this part to adjust the keys in state_dict
            model.load_state_dict(state_dict, strict=True)
        return model
        
    
        # Load the state dict from the file
    if jit_model:
        model = torch.jit.load(modelweight_path)
        print('loaded jit model')
        return model
        
    else:
        checkpoint = torch.load(modelweight_path)  
    
        if 'state_dict' in checkpoint:
            
            # Restore the model state_dict
            state_dict = checkpoint['state_dict']
            model = wrap_model(state_dict, model)
            if train:

                
                optimiser=get_optimiser(model)
                # Restore the optimizer state_dict
                optimiser.load_state_dict(checkpoint['optimizer'])

                # Restore the scaler state_dict (if you are using automatic mixed precision)
                scaler.load_state_dict(checkpoint['scaler'])

                # Restore the learning rate scheduler state_dict
                lr_scheduler.load_state_dict(checkpoint['scheduler'])

                # Optionally, restore the epoch
                start_epoch = checkpoint['epoch']
                # loss = checkpoint['loss']

                print(f"Model, optimizer, scaler, and scheduler states have been restored from epoch {start_epoch}")              
            
                return model,optimiser,scaler,lr_scheduler,start_epoch
            else:
               return model 
        else:        
            model = wrap_model(checkpoint,model)
   

        return model
    
def model_loader_ind(modelweight_path,model,train=False,optimiser=None,scaler=None,lr_scheduler=None,start_epoch=None):

    # Load the state dict from the file
    checkpoint = torch.load(modelweight_path)
    
    if 'state_dict' in checkpoint:
        
        # Restore the model state_dict
        state_dict = checkpoint['state_dict']
        if train:
            # Restore the optimizer state_dict
            optimiser.load_state_dict(checkpoint['optimizer'])

            # Restore the scaler state_dict (if you are using automatic mixed precision)
            scaler.load_state_dict(checkpoint['scaler'])

            # Restore the learning rate scheduler state_dict
            lr_scheduler.load_state_dict(checkpoint['scheduler'])

            # Optionally, restore the epoch
            start_epoch = checkpoint['epoch']

            print(f"Model, optimizer, scaler, and scheduler states have been restored from epoch {start_epoch}")
    else:
        state_dict = checkpoint
    
    # Check if the state dict contains keys prefixed with 'module.'
    # This indicates that the model was saved with DataParallel
    is_dataparallel = any(key.startswith('module.') for key in state_dict.keys())

    if is_dataparallel:
        # Wrap the model with DataParallel before loading the state dict
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict, strict=True)
    else:
        # If there's no 'module.' prefix, load the state dict as is
        # This also handles the case where the model needs to be wrapped but the saved model wasn't
        # If necessary, you can modify this part to adjust the keys in state_dict
        model.load_state_dict(state_dict, strict=True)

    return model#,optimiser,scaler,lr_scheduler,start_epoch

def get_model_paths(modelweight_folder_path):
    modelnames=sorted(os.listdir(modelweight_folder_path)) # assumes only cluster models here

    # modelpaths=sorted([os.path.join(modelweight_folder_path,x) for x in modelnames if 'Cluster' in x]) 
    modelpaths=sorted([os.path.join(modelweight_folder_path,x) for x in modelnames if '_j' in x])
    return modelpaths  

    
def sum_weights_from_file(file_path):
    state_dict = torch.load(file_path)
    # Check if the loaded dictionary is a checkpoint
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    total_weight_sum = sum(torch.sum(param).item() for param in state_dict.values())
    return total_weight_sum

def lesionwise_metrics(test_outputs,test_labels, sub_id):
    df= lesion_wise(test_outputs[0].cpu().numpy(),test_labels[0].cpu().numpy(),output='./lesion_wise.csv')
    df_row = df.stack().to_frame().T
    #columns are tuples after stacking, joining the row and col names with an underscore
    df_row.columns=['_'.join(map(str,col)).strip() for col in df_row.columns.values]
  
    df_row['Subject_ID'] = sub_id
    new_order = ['Subject_ID'] + [col for col in df_row.columns if col!='Subject_ID']
    df_row=df_row[new_order]
    return df_row
    
def dropout_model_list(load_path):
    '''
    Returns a list of models with dropout loaded ready to be called 
    Input: folder path with desired modelweights
    Output: List with model objects 
    '''
    model_list=[]
    seedlist=np.arange(4)
    model = model_loader(load_path)
    print([name for name,param in model.named_parameters()])

    for seed in seedlist:
        torch.manual_seed(seed)
        model = model_loader(load_path)
        model_list.append(dropout_network(model))
    return model_list
    
def model_in_list(modelweight_folder_path):
    '''
    Returns a list of models loaded ready to be called 
    Input: folder path with desired modelweights
    Output: List with model objects 
    '''
    modelnames=get_model_paths(modelweight_folder_path)
    modelpaths=[os.path.join(modelweight_folder_path,x) for x in modelnames]
    model_list=[model_loader(x) for x in modelpaths]
    return model_list
    
def cluster_model_in_list(modelweight_folder_path):
    '''
    Returns a list of models loaded ready to be called 
    Input: folder path with desired modelweights
    Output: List with model objects 
    '''
    modelnames=sorted(os.listdir(modelweight_folder_path)) # assumes only cluster models here
    
    modelpaths=[os.path.join(modelweight_folder_path,x) for x in modelnames if 'Cluster' in x]
    # print(modelpaths)
    # for idx, modelpath in enumerate(modelpaths):
        # total_weight_sum = sum_weights_from_file(modelpath)
        # print(f"Model {idx} total weight sum: {total_weight_sum}")
    model_list=[model_loader(x) for x in modelpaths]
    return model_list

def load_cluster_centres(cluster_path='/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers'):
    '''
    Load cluster centres
    '''
     
    centre_names=[f'{raw_features_filename}_centre_{i}.npy' for i in range(4)]
    # centre_names=[x for x in centre_names if 'cluster' in x]
    # print(type(centre_names[0]))
    cluster_centre_paths=[os.path.join(cluster_path,x) for x  in centre_names]
    print(cluster_centre_paths)
    cluster_centres = [np.load(x) for x in cluster_centre_paths]
    max_bound=np.load(f'/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers/{raw_features_filename}_max_bound.npy')
    # print(max_bound.shape, 'max bound shape')
    min_bound=np.load(f'/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers/{raw_features_filename}_min_bound.npy')
    return cluster_centres, min_bound,max_bound

def extract_features(test_input,base_model):
    '''
    Switching logic do decide between various feature extraction options
    '''
    if '5x' in cluster_files:
        if model_name== 'transformer':
            print('Not implemented yet. Please implement the function if necessary inside encoded_features_x5.py')
        else:
            feat = single_encode_x5(test_input,base_model) # assuming no need to collapse batch dimension     
    else:
        if model_name== 'transformer':
            feat = single_encode_tfmr(test_input,base_model)
        else:
            feat = single_encode(test_input,base_model)   
    return feat
    
    
    
def model_selector(model_list,test_data,base_model,min_bound,max_bound,cluster_centres,dist_lists):
    ''' Calculates features from test samples then calculates distances from saved cluster
        centres. Following that makes the evaluation using the closest model.'''
    #select a mode and perform a single evaluation
    test_input=test_data["image"].to(device)
    sub_id = test_data["id"][0][-9:]
    print('sub_id',sub_id)
    base_model.eval()
    # base_model_weight_sum = sum(torch.sum(param).item() for param in base_model.state_dict().values())
    # print('base model weight sum', base_model_weight_sum)
    
    feat = extract_features(test_input,base_model)  
    
    normalised_feat=(feat-min_bound)/(max_bound-min_bound)
    
    # distances=[distance.euclidean(feat,x) for x in cluster_centres]
    distances=[]
    for center in cluster_centres:
        print(normalised_feat.mean(), center.mean(), min_bound.mean(), max_bound.mean())
        distance = np.linalg.norm(normalised_feat-center) 
        distances.append(distance)
        print(distance)
   
    # distances=[distance.cosine(feat,x) for x in cluster_centres]
    # print(distances)
    model_index=np.argmin(distances) # assuming everything is sorted alphabetically as intent
    
    # print(centre_names[model_index])
    dist_lists[sub_id]= distances +([distances[model_index]]) # extending the list
    model_selected=model_list[model_index]
    
    model_selected.eval()
    # base_dice, base_tc,base_wt,base_et, base_test_outputs, base_test_labels=eval_single(base_model,test_data)
    
    current_dice, tc,wt,et, test_outputs, test_labels=eval_single(model_selected,test_data)   
    return current_dice, tc,wt,et, test_outputs, test_labels,model_index,dist_lists,sub_id

    
def eval_model_selector(modelweight_folder_path, loader):
    '''
    Evaluate using the closest cluster expert model    
    '''
   # Returns a list of models loaded ready to be called
    model_list = cluster_model_in_list(modelweight_folder_path)
    
    # for idx, model in enumerate(model_list):
        # print(f"Model {idx} address: {id(model)}") # to print memory address model is stored at
    # for idx, model in enumerate(model_list):
        # print(f"Model {idx} weights: {next(iter(model.state_dict().values()))[:5]}") 
    for idx, model in enumerate(model_list):
        total_weight_sum = sum(torch.sum(param).item() for param in model.state_dict().values())
        print(f"Model {idx} total weight sum: {total_weight_sum}")
    # sys.exit()
    base_model = model_loader(encoder_path,model_name)
    base_model.eval()
    base_model_weight_sum = sum(torch.sum(param).item() for param in base_model.state_dict().values())
    print('base model weight sum', base_model_weight_sum)
    cluster_centres,min_bound,max_bound = load_cluster_centres()
    
    print([x.mean() for x in cluster_centres])
    ind_scores = dict()
    slice_dice_metric = DiceMetric(include_background=True, reduction='none')
    slice_dice_scores = dict()
    slice_pred_area = dict()
    slice_gt_area = dict()
    model_closest={}
    
    
    dist_lists={}
    result_rows=[]
    with torch.no_grad():
        for i,test_data in enumerate(loader):
            
            current_dice, tc,wt,et, test_outputs, test_labels,model_index,dist_lists,sub_id=model_selector(model_list,test_data,base_model,min_bound,max_bound,cluster_centres,dist_lists)  
            df_row=lesionwise_metrics(test_outputs,test_labels, sub_id)
            result_rows.append(df_row)
            model_closest[sub_id]=model_index
            dice_metric(y_pred=test_outputs, y=test_labels)
            dice_metric_batch(y_pred=test_outputs, y=test_labels)
            ind_scores[sub_id] = {'average':round(current_dice,4), 'tc':round(tc,4),'wt':round(wt,4),'et':round(et,4)}
            metrics = dice_profile_and_sphericity(test_outputs[0])  # 
            # Average the metrics across channels and views
            
            avg_dice = np.mean([metrics[view][channel]['Dice'] for view in ['Sagittal', 'Frontal', 'Axial'] for channel in ['TC', 'WT', 'ET']])
            avg_delta_area = np.mean([metrics[view][channel]['dArea'] for view in ['Sagittal', 'Frontal', 'Axial'] for channel in ['TC', 'WT', 'ET']])
            avg_regularity = np.mean([metrics[view][channel]['Regularity'] for view in ['Sagittal', 'Frontal', 'Axial'] for channel in ['TC', 'WT', 'ET']])

            # Update the ind_scores dictionary
            ind_scores[sub_id].update({
                
                'avg_dice_profile': round(avg_dice, 4),
                'avg_delta_area': round(avg_delta_area, 4),
                'avg_regularity': round(avg_regularity, 4)
            })
        metric_org = dice_metric.aggregate().item()        
        metric_batch_org = dice_metric_batch.aggregate()
        all_results = pd.concat(result_rows)
        dice_metric.reset()
        dice_metric_batch.reset()
    metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()
    print("Metric on original image spacing: ", metric_org)
    print(f"metric_tc: {metric_tc:.4f}", f"   metric_wt: {metric_wt:.4f}", f"   metric_et: {metric_et:.4f}")
    return metric_org,metric_tc,metric_wt,metric_et,ind_scores,model_closest,dist_lists,slice_dice_scores,slice_gt_area,slice_pred_area,all_results
    
# New function to calculate weights from distances
def calculate_weights(distances):
    inverted_distances = [1/d if d != 0 else 0 for d in distances]
    total = sum(inverted_distances)
    normalized_weights = [d/total for d in inverted_distances]
    return normalized_weights

def calculate_squared_weights(distances):
    inverted_distances = [1/(d**2) if d != 0 else 0 for d in distances]
    total = sum(inverted_distances)
    normalized_weights = [d/total for d in inverted_distances]
    return normalized_weights
    
TT_transforms=[
# # CenterSpatialCrop(roi_size=(192,192,144)),
# CropForeground(),
# AdjustContrast(gamma=1.1),
# Identity(),
# ScaleIntensity(factor=0.1),
AdjustContrast(gamma=0.5),
AdjustContrast(gamma=0.25),
AdjustContrast(gamma=1),
AdjustContrast(gamma=1.25),
]
fix_size=SpatialPad(spatial_size=(240,240,155))


def modelweight_sums(modelpaths):
    sums=[]
    for idx, modelpath in enumerate(modelpaths):
        total_weight_sum = sum_weights_from_file(modelpath)
        print(f"Model {idx} total weight sum: {total_weight_sum}")  # checking to see models are unique
        sums.append(total_weight_sum)
    return sums
    
def ensemble_inference(modelweight_folder_path,loader):
    # Returns a list of models loaded ready to be called
    model_list = model_in_list(modelweight_folder_path)
    
    with torch.no_grad():
        for i,test_data in enumerate(loader):
            sub_id=test_data["id"][0]
            test_outputs_i=[[] for x in model_list] 
            test_inputs = test_data["image"].to(device)
            for j,model in enumerate(model_list):
                model.eval()
                test_data["pred"]=inference(test_inputs,model)
                decollated_raw = decollate_batch(test_data)
                test_outputs_i[j] = decollated_raw[0]["pred"] #assuming batch size of 1
            final_sum = 0
            for pred in test_outputs_i:
                final_sum+=pred
            # assigning to 0th item in batch assuming bs=1
            test_data["pred"][0] = final_sum/len(test_outputs_i)
            test_data=[post_trans_test(i) for i in decollate_batch(test_data)]
            test_outputs=from_engine(["pred"])(test_data) # create a list of 'pred' samples 
            
            nampendix=modelweight_folder_path.split('/')[-1]
            #returns a list of tensors
            new_dir = os.path.join(output_path, f'b23Ensemble_{nampendix}')
            os.makedirs(new_dir, exist_ok=True)
            nii_img=nib.Nifti1Image(test_outputs[0].numpy(),np.eye(4))
            ##Save the image to a .nii.gz file
            nii_img.to_filename(f'{new_dir}/{sub_id}.nii.gz')
            
def distance_ensembler(modelweight_folder_path, loader,option=DE_option):
    '''
    Evaluate using the closest cluster expert model. If DE_option= 'plain' in config will perform a standard averaging ensemble
    '''
       
    modelpaths = get_model_paths(modelweight_folder_path)
    
    assert len(modelpaths)>0 , 'No paths in modelpath folder, ensure correct path provided'
    print('models loaded from ', modelpaths)
    modelweight_sums(modelpaths)
    if dropout:
        print('''
        ##############################DROPOUT~~~~~~~################~~~~~###############
        ''')
        model_list=dropout_model_list(load_path)
    else:
        model_list=[model_loader(x) for x in modelpaths]
    for idx, model in enumerate(model_list):
        print(f"Model {idx} address: {id(model)}") # to print memory address model is stored at
    for idx, model in enumerate(model_list):
        print(f"Model {idx} weights: {next(iter(model.state_dict().values()))[0][0][0]}") 
  
    base_model=model_loader(encoder_path,model_name)
    base_model.eval()
    cluster_path='/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers'
    centre_names=sorted(os.listdir(cluster_path))
    centre_names=[x for x in centre_names if f'{raw_features_filename}_centre' in x]
    print(type(centre_names[0]))
    cluster_centre_paths=[os.path.join(cluster_path,x) for x  in centre_names]
    print(sorted(cluster_centre_paths))
    cluster_centres=[np.load(x) for x in sorted(cluster_centre_paths)]
    
    print([x.mean() for x in cluster_centres])
    ind_scores = dict()
    slice_dice_metric = DiceMetric(include_background=True, reduction='none')
    slice_dice_scores = dict()
    slice_pred_area = dict()
    slice_gt_area = dict()
    model_closest={}
    
   #creating empty list of lists to store individual values separately for each model we then iterate through each model to fill these 
    dist_lists={}
    result_rows=[]
    with torch.no_grad():
        for i,test_data in enumerate(loader):
            current_dice_i, tc_i,wt_i,et_i, test_outputs_i, test_labels_i,decollated_raw = [[] for x in modelpaths],[[] for x in modelpaths],[[] for x in modelpaths],[[] for x in modelpaths],[[] for x in modelpaths],[[] for x in modelpaths],[[] for x in modelpaths]
            test_input=test_data["image"].to(device)
            sub_id = test_data["id"][0][-9:]
            print('sub_id',sub_id)
            
            feat = extract_features(test_input,base_model)
            
            normalised_feat=(feat-min_bound)/(max_bound-min_bound)
            
 
            distances=[np.linalg.norm(normalised_feat-center) for center in cluster_centres]
            if DE_option=='squared':
                ensemble_weights=calculate_squared_weights(distances)            
            else:
                ensemble_weights=calculate_weights(distances)

           
            model_index=np.argmin(distances) # assuming everything is sorted alphabetically as intent
            model_closest[sub_id]=model_index
            # print(centre_names[model_index])
            dist_lists[sub_id]= distances +([distances[model_index]])
            
            # base_dice, base_tc,base_wt,base_et, base_test_outputs, base_test_labels=eval_single(base_model,test_data)
            test_inputs = test_data["image"].to(device)
            # print(len(ensemble_weights),'ensemble_weights')
            decollated_raw_list=[]
            for j in range(len(model_list)): 
                # print('we are in counter ', j)
                if TTA_ensemble:
                    
                    transformed_input = TT_transforms[j](test_inputs[0]) #pointing to first item in batch- will only work with bs=1
                    # print('the transformed input shape is', transformed_input.shape)
                    transformed_input =transformed_input.unsqueeze(dim=0) # restoring the batch dimension for inference
                    # print('the transformed input shape is after unsquezing', transformed_input.shape)
                    test_data["pred"] = inference(transformed_input,base_model)
                    # print('pred,shape is', test_data["pred"].shape)
                    # if j<2:
                    test_data["pred"]=fix_size(test_data["pred"][0]).unsqueeze(dim=0)
                        # print('pred,shape is after fixing shape', test_data["pred"].shape)
           
                   
                else:  
                    try:
                        model_selected = model_list[j]          
                        model_selected.eval()
                    except:
                        print(f'NOT possible to load model {j+1}')
                        sys.exit()
                        continue
                    test_data["pred"] = inference(test_inputs,model_selected)
                

                if option=='plain':
                    decollated_raw = decollate_batch(test_data)#[{**x, "pred": x["pred"]} for x in decollate_batch(test_data)]
                    test_outputs_i[j] = decollated_raw[0]["pred"] 
                    test_labels_i[j] = decollated_raw[0]["mask"]
                    
                    trans_out = post_trans(decollated_raw[0])
                    output = [trans_out["pred"].to(device)]
                    label = [trans_out["mask"].to(device)]
                    dice_metric_ind(y_pred= output, y=label)
                    current_dice_i[j] = dice_metric_ind.aggregate(reduction=None).item()
                    print(f'Dice score for model {j} is {current_dice_i[j]}')
                    dice_metric_ind.reset()
                    
                else:                    
                    decollated_raw=[{**x, "pred": x["pred"] * ensemble_weights[j]} for x in decollate_batch(test_data)] #list of unthresholded outputs in batch - 1 item if batch size is 1 - decollated raw should be a list of lists and each list has one dict
                decollated_raw_list.append(decollated_raw)
                # decollated_raw[j]=[post_trans(x) for x in decollate_batch(test_data)]
            combined_dict_list=[]
            weighted_sum={}
            
            
            for ii, a in enumerate(decollated_raw_list):
                #now we have 4 decollated lists of raw dictionaries representing output from same batch for the 4 models 
 
               
                if len(a)>0:
                    if type(a[0])==list:
                        print('batch greater than 1,exiting')
                        sys.exit()
                      
                    else:                    
                        if ii==0:
                            weighted_sum["pred"]=a[0]["pred"]
                            weighted_sum["mask"]=a[0]["mask"]
                        else:
                            weighted_sum["pred"]+=a[0]["pred"]
                else:
                    print('NOT MUCH TO DO @@@')
                        
            if DE_option=='plain':                         
                weighted_sum["pred"]=weighted_sum["pred"]/len(decollated_raw)

            combined_dict_list.append(weighted_sum) # assuming batch size=1
            
            tested_data= post_trans(weighted_sum) #[post_trans(ii) for ii in combined_dict_list] 
           
            test_outputs,test_labels = tested_data["pred"],tested_data["mask"]
            test_outputs = [tested_data["pred"].to(device)]# [tensor.to(device) for tensor in test_outputs]
            test_labels = [tested_data["mask"].to(device)]#[tensor.to(device) for tensor in test_labels]
            #from_engine(["pred","mask"])(tested_data)# returns two lists of tensors
            df= lesion_wise(test_outputs[0].cpu().numpy(),test_labels[0].cpu().numpy(),output='./lesion_wise.csv')
            df_row = df.stack().to_frame().T
            #columns are tuples after stacking, joining the row and col names with an underscore
            df_row.columns=['_'.join(map(str,col)).strip() for col in df_row.columns.values]
          
            df_row['Subject_ID'] = sub_id
            new_order = ['Subject_ID'] + [col for col in df_row.columns if col!='Subject_ID']
            df_row=df_row[new_order]
            
            
            dice_metric_ind(y_pred=test_outputs, y=test_labels)
            dice_metric_ind_batch(y_pred=test_outputs, y=test_labels)
            current_dice = dice_metric_ind.aggregate(reduction=None).item()
            print(current_dice)
            batch_ind = dice_metric_ind_batch.aggregate()
            tc,wt,et = batch_ind[0].item(),batch_ind[1].item(),batch_ind[2].item()
            dice_metric_ind.reset()
            dice_metric_ind_batch.reset()
            
            dice_metric(y_pred=test_outputs, y=test_labels)
            dice_metric_batch(y_pred=test_outputs, y=test_labels)
            ind_scores[sub_id] = {'average':round(current_dice,4), 'tc':round(tc,4),'wt':round(wt,4),'et':round(et,4)}
            df_row['original index'] = test_indices[i]
            df_row['Plain_dice_avg'] = round(current_dice,4)
            df_row['pd_tc'] = round(tc,4)
            df_row['pd_wt'] = round(wt,4)
            df_row[ 'pd_et'] = round(et,4) 
            result_rows.append(df_row)
            metrics = dice_profile_and_sphericity(test_outputs[0])  # Assuming test_outputs[0] is your mask_input
            # Average the metrics across channels and views
            dice_prof_vals=[metrics[view][channel]['Dice'] for view in ['Sagittal', 'Frontal', 'Axial'] for channel in ['TC', 'WT', 'ET']]
            da_vals=[metrics[view][channel]['dArea'] for view in ['Sagittal', 'Frontal', 'Axial'] for channel in ['TC', 'WT', 'ET']]
            reg_vals = [metrics[view][channel]['Regularity'] for view in ['Sagittal', 'Frontal', 'Axial'] for channel in ['TC', 'WT', 'ET']]
            avg_dice = np.mean([dice_prof_vals[x] for x in np.nonzero(dice_prof_vals)[0].astype(int)])
            avg_delta_area = np.mean([da_vals[x] for x in np.nonzero(da_vals)[0].astype(int)])
            avg_regularity = np.mean([reg_vals[x] for x in np.nonzero(reg_vals)[0].astype(int)])

            # Update the ind_scores dictionary
            ind_scores[sub_id].update({
                
                'avg_dice_profile': round(avg_dice, 4),
                'avg_delta_area': round(avg_delta_area, 4),
                'avg_regularity': round(avg_regularity, 4)
            })
        metric_org = dice_metric.aggregate().item()        
        metric_batch_org = dice_metric_batch.aggregate()
        all_results = pd.concat(result_rows)
        dice_metric.reset()
        dice_metric_batch.reset()
    metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()
    print("Metric on original image spacing: ", metric_org)
    print(f"metric_tc: {metric_tc:.4f}", f"   metric_wt: {metric_wt:.4f}", f"   metric_et: {metric_et:.4f}")
    return metric_org,metric_tc,metric_wt,metric_et,ind_scores,model_closest,dist_lists,slice_dice_scores,slice_gt_area,slice_pred_area,all_results

def eval_single(loaded_model,test_data):
    '''
    Takes an initialised model and returns the dice score
    '''
    with torch.no_grad():
        loaded_model.eval()
        #assuming model eval already set
        print(f"Model address: {id(loaded_model)}") #printing mem address to ensure different models
        test_inputs = test_data["image"].to(device) # pass to gpu
        
        test_data["pred"] = inference(test_inputs,loaded_model)
        test_data=[post_trans(ii) for ii in decollate_batch(test_data)] #returns a list of n dicts where n=batch_size
        test_outputs,test_labels = from_engine(["pred","mask"])(test_data)# returns two lists of tensors
        test_outputs = [tensor.to(device) for tensor in test_outputs]
        test_labels = [tensor.to(device) for tensor in test_labels]
        dice_metric_ind(y_pred=test_outputs, y=test_labels)
        dice_metric_ind_batch(y_pred=test_outputs, y=test_labels)
        current_dice = dice_metric_ind.aggregate(reduction=None).item()
        print(current_dice)
        batch_ind = dice_metric_ind_batch.aggregate()
        tc,wt,et = batch_ind[0].item(),batch_ind[1].item(),batch_ind[2].item()
        dice_metric_ind.reset()
        dice_metric_ind_batch.reset()
    return current_dice, tc,wt,et, test_outputs, test_labels
    
def eval_single_raw(loaded_model,test_data):
    '''
    Takes an initialised model and returns the dice score
    '''
    with torch.no_grad():
        loaded_model.eval()
        #assuming model eval already set
        # print(f"Model address: {id(loaded_model)}") #printing mem address to ensure different models
        test_inputs = test_data["image"].to(device) # pass to gpu
        
        test_outputs = inference(test_inputs,loaded_model)
      
    return test_outputs
    
def plot_expert_performance(ind_score_df,save_path,plot_together=True):

    now=datetime.now().strftime('%Y-%m-%d_%H-%M')
    # save_in=os.path.join(save_path,now)
    # os.makedirs(save_in,exist_ok=True)
    if plot_together==True:
        ind_score_df=ind_score_df.sort_values(by='Base Average Dice',ignore_index=True)
        
        #setting a figure with plt allows the size to be fixed
        plt.figure(figsize=(12,6))
        plt.rcParams.update({'font.size': 18})
        #now we plot all desired values in sequence
        plt.plot(ind_score_df['average'],color='red',label='Expert Average')
        plt.plot(ind_score_df['Base Average Dice'],color='red',linestyle=':',linewidth=2,label='Baseline Average')
      
        ind_score_df=ind_score_df.sort_values(by='Base TC',ignore_index=True)
        plt.plot(ind_score_df['tc'],color='black',label='Expert TC')
        plt.plot(ind_score_df['Base TC'],color='black', linestyle=':',linewidth=2,label='Baseline TC')
       
        
        ind_score_df=ind_score_df.sort_values(by='Base WT',ignore_index=True)
        plt.plot(ind_score_df['wt'],color='green',label='Expert WT')
        plt.plot(ind_score_df['Base WT'],color='green', linestyle=':', linewidth=2,label='Baseline WT')
       
        ind_score_df=ind_score_df.sort_values(by='Base ET',ignore_index=True)
        plt.plot(ind_score_df['et'],color='pink',label='Expert ET')
        plt.plot(ind_score_df['Base ET'], color='pink', linestyle=':',linewidth=2,label='Baseline ET')
        
        plt.legend()
        plt.grid(which='both')
        plt.xlabel('Samples')
        plt.ylabel('Dice Score')
        # plt.title('Comparison of Model Performances')
        # plot_name=os.path.join(save_in,'indComp.svg')
        plt.savefig(save_path)
    
    else:
        #to sort the data frame by values we do
        ind_score_df=ind_score_df.sort_values(by='Base Average Dice',ignore_index=True)
        
        #setting a figure with plt allows the size to be fixed
        plt.figure(figsize=(12,6))
        
        #now we plot all desired values in sequence
        plt.plot(ind_score_df['average'],color='red',label='Expert Average')
        plt.plot(ind_score_df['Base Average Dice'],color='red',linestyle=':',linewidth=2,label='Baseline Average')
        plot_name=os.path.join(save_in,'Average.jpg')
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('Dice Score')
        plt.title('Comparison of Model Performances')
        plt.savefig(plot_name)
        #TC now
        plt.figure(figsize=(12,6))
        ind_score_df=ind_score_df.sort_values(by='Base TC',ignore_index=True)
        plt.plot(ind_score_df['tc'],color='black',label='Expert TC')
        plt.plot(ind_score_df['Base TC'],color='black', linestyle=':',linewidth=2,label='Baseline TC')
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('Dice Score')
        plt.title('Comparison of Model Performances')
        plot_name=os.path.join(save_in,'TC.jpg')
        plt.savefig(plot_name)
        
        plt.figure(figsize=(12,6))
        ind_score_df=ind_score_df.sort_values(by='Base WT',ignore_index=True)
        plt.plot(ind_score_df['wt'],color='green',label='Expert WT')
        plt.plot(ind_score_df['Base WT'],color='green', linestyle=':', linewidth=2,label='Baseline WT')
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('Dice Score')
        plt.title('Comparison of Model Performances')
        plot_name=os.path.join(save_in,'WT.jpg')
        plt.savefig(plot_name)
        
        plt.figure(figsize=(12,6))
        ind_score_df=ind_score_df.sort_values(by='Base ET',ignore_index=True)
        plt.plot(ind_score_df['et'],color='pink',label='Expert ET')
        plt.plot(ind_score_df['Base ET'], color='pink', linestyle=':',linewidth=2,label='Baseline ET')
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('Dice Score')
        plt.title('Comparison of Model Performances')
        plot_name=os.path.join(save_in,'ET.jpg')
        plt.savefig(plot_name)
    
    
    return f'plot saved successully' #in {save_in}' 
    
def plot_scatter(dice, gt, pred, save_path, use_slice_number=True, min_size=10, max_size=100):

    ''' expected shape for dice: [number_of_slices, samples]
    .values extracts just the values and not the slice numbers
    ie 76 arrays of 144 dice values representing each slice
    '''
    # Flatten the dataframes to get values for all slices across all samples
    dice_scores = dice.values.flatten()
    gt_area = gt.values.flatten()
    pred_area = pred.values.flatten()
    
    
    # Determine color based on whether prediction area is greater or less than ground truth
    colors = ['red' if p > g else 'blue' if p < g else 'green' for p, g in zip(pred_area, gt_area)]
    

    # Normalize the sizes
    scaler = MinMaxScaler(feature_range=(min_size, max_size))
    sizes = scaler.fit_transform(gt_area.reshape(-1, 1))
    
    plt.figure(figsize=(12, 7))
    
    if use_slice_number:
        # Slice number on x-axis
        x_values = np.tile(np.arange(dice.shape[0]), dice.shape[1])
        # colors = colors[:len(x_values)]
        # sizes = sizes[:len(x_values)].flatten()
        plt.scatter(x_values, dice_scores, c=colors, s=sizes, alpha=0.5)
        plt.xlabel('Slice Number')
    else:
        # Samples on x-axis (essentially sample order)
        x_values = np.repeat(np.arange(dice.shape[1]), dice.shape[0])
        plt.scatter(x_values, dice_scores, c=colors, s=sizes, alpha=0.5)
        plt.xlabel('Sample')
    
    plt.ylabel('Dice Score')
    plt.title('Dice Scores vs Slices/Samples')
    plt.grid(True)
    
    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()
        
def plot_prediction(image_slice, gt_mask, pred_mask, o_path, title, plot_slice):
    tp = len(np.nonzero(pred_mask & gt_mask)[0])
    fp = len(np.nonzero(pred_mask & ~gt_mask)[0])
    fn = len(np.nonzero(~pred_mask & gt_mask)[0])
    den = 2 * tp + fp + fn
    if plot_slice:
        if den > 0:
            dice_value = 2 * tp / den
            dice = f" - Dice Score: {dice_value:.4f}"  # For 4 decimal places
            fig, ax = plt.subplots(2, 2, figsize=(20, 20))

            ax[0][0].imshow(image_slice[0], cmap='gray', origin='lower')
            ax[0][0].set_title('Original Slice', fontsize=40)
            combined_mask = np.zeros_like(pred_mask, dtype=int)
            combined_mask[pred_mask & gt_mask] = 1  # True positives - green in cm 
            combined_mask[pred_mask & ~gt_mask] = 2  # False positives - red in cm
            combined_mask[~pred_mask & gt_mask] = 3  # False negatives - blue in cm
            print(title, np.unique(combined_mask))
            
            colors = [(0, 0, 0, 0), (0, 1, 0, 0.3), (1, 0, 0, 0.3), (0, 0, 1, 0.3)]
            cm = ListedColormap(colors)
            if not np.any(gt_mask):  # not ideal but better than nothing
                colors = [(0, 0, 0, 0), (0, 1, 0, 0.3), (1, 0, 0, 0.3), (1, 0, 0, 0.3)]
            cm = ListedColormap(colors)
            gt_colors = [(0, 0, 0, 0), (0, 1, 0, 0.3)]
            gm = ListedColormap(gt_colors)
            pred_colors = [(0, 0, 0, 0), (1, 0, 0, 0.3)]
            pm = ListedColormap(pred_colors)
            
            ax[0][1].imshow(image_slice[1], cmap='gray', origin='lower')
            ax[0][1].imshow(combined_mask, cmap=cm, origin='lower')
            ax[0][1].set_title(title , fontsize=40)

            ax[1][0].imshow(image_slice[2], cmap='gray', origin='lower')
            ax[1][0].imshow(gt_mask, cmap=gm, origin='lower')
            ax[1][0].set_title("Ground Truth" , fontsize=40)

            ax[1][1].imshow(image_slice[3], cmap='gray', origin='lower')
            ax[1][1].imshow(pred_mask, cmap=pm, origin='lower')
            ax[1][1].set_title("Prediction" + dice, fontsize=40)

            for a in ax.ravel():
                a.axis('off')  # Turn off axis labels and ticks

            plt.tight_layout()
            plt.savefig(os.path.join(o_path, title + dice + '.png'))
            plt.close()
    else:
        dice = " - Dice Score: Nan"

    
    

    
def find_centroid(mask_channel):
    """
    Compute the centroid slice indices along each view for a given mask channel.
    """
    coords = np.argwhere(mask_channel)
    return coords.mean(axis=0).astype(int)  
    


def calculate_perimeter_and_regularity(mask_slice, kernel):
    area_slice = torch.nonzero(mask_slice).numel()
    perimeter = torch.nonzero(torch.tensor(cv2.erode(mask_slice.cpu().numpy(), kernel))).numel()
    rep_rad = np.sqrt(area_slice / np.pi)
    reg_score = perimeter / (2 * np.pi * rep_rad + 0.001)
    return area_slice, reg_score

def calculate_dice_score(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    dice=(2 * intersection) / (mask1.sum() + mask2.sum() + 0.001)

    return dice.cpu().numpy()
    
def calculate_sphericity(mask):
    channels = ['TC', 'WT', 'ET']
    pred_feat = {}
    for o in range(mask.shape[0]):
        mask_volume = torch.nonzero(mask[o]).size(0)
        pi = torch.tensor(np.pi)
        rep_rad = torch.pow((0.75 * mask_volume / pi), (1/3))
        sphericity = mask_volume / ((4/3) * pi * torch.pow(rep_rad, 3))
        pred_feat['Sphericity_[channels[o]}']=sphericity
    return pred_feat

def dice_profile_and_sphericity(mask_input):
    kernel = np.ones((3, 3), np.uint8)
    dims = mask_input.shape[1:]  # H, W, D dimensions
    channels = ['TC', 'WT', 'ET']
    metrics = {}

    # Initializing metrics for each dimension and channel
    for dim_label in ['Sagittal', 'Frontal', 'Axial']:
        metrics[dim_label] = {ch: {'Area': [], 'Regularity': [], 'Dice': [], 'dArea': []} for ch in channels}

    # Iterate over each dimension
    for dim in range(3):
        for i in range(dims[dim]): #eg dims[0]==192
            # Select the appropriate 2D slices for each channel
            if dim == 0:  # Sagittal
                slices = mask_input[:, i, :, :]
            elif dim == 1:  # Frontal
                slices = mask_input[:, :, i, :]
            else:  # Axial
                slices = mask_input[:, :, :, i]#.squeeze(3)

            # Iterate over each channel
            for ch in range(mask_input.shape[0]):
                slice = slices[ch]
                prev_slice = None
                if i > 0:
                    if dim == 0:
                        prev_slice = mask_input[ch, i-1, :, :]#.squeeze(1)
                    elif dim == 1:
                        prev_slice = mask_input[ch, :, i-1, :]#.squeeze(2)
                    else:
                        prev_slice = mask_input[ch, :, :, i-1]#.squeeze(3)

                # Calculate area and regularity and append each slice value to a list
                area_slice, reg_score = calculate_perimeter_and_regularity(slice, kernel)
                metrics['Sagittal' if dim == 0 else 'Frontal' if dim == 1 else 'Axial'][channels[ch]]['Area'].append(area_slice)
                metrics['Sagittal' if dim == 0 else 'Frontal' if dim == 1 else 'Axial'][channels[ch]]['Regularity'].append(reg_score)

                # Calculate Dice scores and dArea if not the first slice
                if prev_slice is not None:
                    dice_score = calculate_dice_score(prev_slice, slice)
                    metrics['Sagittal' if dim == 0 else 'Frontal' if dim == 1 else 'Axial'][channels[ch]]['Dice'].append(dice_score)
                    prev_area = metrics['Sagittal' if dim == 0 else 'Frontal' if dim == 1 else 'Axial'][channels[ch]]['Area'][-2] #-2 selecting the 2nd to last area item
                    darea = 2 * abs(area_slice - prev_area) / (area_slice + prev_area + 0.001)
                    metrics['Sagittal' if dim == 0 else 'Frontal' if dim == 1 else 'Axial'][channels[ch]]['dArea'].append(darea)

    # Averaging the lists in the metrics
    for dim_label in ['Sagittal', 'Frontal', 'Axial']:
        for ch in channels:
            for key in metrics[dim_label][ch]:
                # print(key,type(metrics[dim_label][ch][key][0]))
                v=metrics[dim_label][ch][key]
                metrics[dim_label][ch][key] = np.mean([v[x] for x in np.nonzero(v)[0].astype(int)]) if v else 0
                #take a non-zero mean instead of mean of all cases- useful when some tumor categories don't exist
    return metrics