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
from Input.config import load_path,VAL_AMP,roi
from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine
from monai.data import DataLoader,decollate_batch
import pandas as pd
import copy

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind = DiceMetric(include_background=True, reduction="mean")
dice_metric_ind_batch = DiceMetric(include_background=True, reduction="mean_batch")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
features_df=pd.read_csv('/scratch/a.bip5/BraTS/trainFeatures_17thJan.csv')
features_df=features_df.drop(columns=['Unnamed: 0','mask_path','subject_id'],errors='ignore') 
max_bound=np.load('/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers/max_bound.npy')
min_bound=np.load('/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers/min_bound.npy')

seed=0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
device = torch.device("cuda:0")
    
def inference(input,model):

    def _compute(input,model):
        
        return sliding_window_inference(
            inputs=input,
            roi_size=roi,
            sw_batch_size=1,
            predictor=model,
            overlap=0,
            )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input,model)
    else:
        return _compute(input,model)
        
def model_loader(modelweight_path):
    '''
    load modelweights regardless of whether it was saved using dataparallel
    '''
    from Training.network import create_model #should hopefully solve the issue
    # model=copy.deepcopy(model)
    model=create_model()
    try:
        model.load_state_dict(torch.load(modelweight_path),strict=True)
        model=torch.nn.DataParallel(model)
    except:
        model=torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(modelweight_path),strict=True)
    return model
    
def sum_weights_from_file(file_path):
    state_dict = torch.load(file_path)
    total_weight_sum = sum(torch.sum(param).item() for param in state_dict.values())
    return total_weight_sum

def eval_model_selector(modelweight_folder_path, loader):
    '''
    Evaluate using the closest cluster expert model
    '''
    modelnames=sorted(os.listdir(modelweight_folder_path)) # assumes only cluster models here
    
    modelpaths=[os.path.join(modelweight_folder_path,x) for x in modelnames if 'Cluster' in x]
    print(modelpaths)
    for idx, modelpath in enumerate(modelpaths):
        total_weight_sum = sum_weights_from_file(modelpath)
        print(f"Model {idx} total weight sum: {total_weight_sum}")
    model_list=[model_loader(x) for x in modelpaths]
    # for idx, model in enumerate(model_list):
        # print(f"Model {idx} address: {id(model)}") # to print memory address model is stored at
    # for idx, model in enumerate(model_list):
        # print(f"Model {idx} weights: {next(iter(model.state_dict().values()))[:5]}") 
    for idx, model in enumerate(model_list):
        total_weight_sum = sum(torch.sum(param).item() for param in model.state_dict().values())
        print(f"Model {idx} total weight sum: {total_weight_sum}")
    # sys.exit()
    base_model=model_loader(load_path)
    base_model.eval()
    cluster_path='/scratch/a.bip5/BraTS/scripts/Evaluation/cluster_centers'
    centre_names=sorted(os.listdir(cluster_path))
    centre_names=[x for x in centre_names if 'cluster' in x]
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
    
    
 
    with torch.no_grad():
        for i,test_data in enumerate(loader):
            test_input=test_data["image"].to(device)
            sub_id = test_data["id"][0][-9:]
            print('sub_id',sub_id)
            feat=single_encode(test_input,base_model) # assuming no need to collapse batch dimension     
            
            
            feat=(feat-min_bound)/(max_bound-min_bound)
            
            # distances=[distance.euclidean(feat,x) for x in cluster_centres]
            distances=[np.linalg.norm(feat-center) for center in cluster_centres]
            
            # distances=[distance.cosine(feat,x) for x in cluster_centres]
            # print(distances)
            model_index=np.argmin(distances) # assuming everything is sorted alphabetically as intent
            model_closest[sub_id]=model_index
            # print(centre_names[model_index])
            model_selected=model_list[model_index]
            
            model_selected.eval()
            # base_dice, base_tc,base_wt,base_et, base_test_outputs, base_test_labels=eval_single(base_model,test_data)
            
            current_dice, tc,wt,et, test_outputs, test_labels=eval_single(model_selected,test_data)   
            
            dice_metric(y_pred=test_outputs, y=test_labels)
            dice_metric_batch(y_pred=test_outputs, y=test_labels)
            ind_scores[sub_id] = {'average':round(current_dice,4), 'tc':round(tc,4),'wt':round(wt,4),'et':round(et,4)}
        metric_org = dice_metric.aggregate().item()        
        metric_batch_org = dice_metric_batch.aggregate()
        dice_metric.reset()
        dice_metric_batch.reset()
    metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()
    print("Metric on original image spacing: ", metric_org)
    print(f"metric_tc: {metric_tc:.4f}", f"   metric_wt: {metric_wt:.4f}", f"   metric_et: {metric_et:.4f}")
    return metric_org,metric_tc,metric_wt,metric_et,ind_scores,model_closest,slice_dice_scores,slice_gt_area,slice_pred_area

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
        test_data=[post_trans(ii) for ii in decollate_batch(test_data)] #returns a list of n tensors where n=batch_size
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
        
def plot_prediction(image_slice, gt_mask, pred_mask, o_path, title,plot_slice):
    

    tp = len(np.nonzero(pred_mask & gt_mask)[0])
    fp = len(np.nonzero(pred_mask & ~gt_mask)[0])
    fn = len(np.nonzero(~pred_mask & gt_mask)[0])
    den = 2*tp + fp + fn
    if plot_slice:
        if den > 0:
            dice_value = 2 * tp / den
            # if dice_value<0.8:
            dice = f" - Dice Score: {dice_value:.4f}"  # For 4 decimal places
            fig, ax = plt.subplots(2, 2, figsize=(20, 20))

            ax[0][0].imshow(image_slice[0], cmap='gray', origin='lower')
            ax[0][0].set_title('Original Slice')
            combined_mask = np.zeros_like(pred_mask, dtype=int)
            combined_mask[pred_mask & gt_mask] = 1  # True positives - green in cm 
            combined_mask[pred_mask & ~gt_mask] = 2  # False positives - red in cm
            combined_mask[~pred_mask & gt_mask] = 3  # False negatives - blue in cm
            print(title, np.unique(combined_mask))
            
            colors = [(0, 0, 0, 0), (0, 1, 0, 0.3),(1, 0, 0, 0.3), (0, 0, 1, 0.3) ]
            cm = ListedColormap(colors)
            if not np.any(gt_mask): #not ideal but better than nothing
                colors = [(0, 0, 0, 0), (0, 1, 0, 0.3),(1, 0, 0, 0.3), (1, 0, 0, 0.3) ]
            cm = ListedColormap(colors)
            gt_colors=[(0,0,0,0),(0,1,0,0.3)]
            gm=ListedColormap(gt_colors)
            pred_colors=[(0,0,0,0),(1,0,0,0.3)]
            pm=ListedColormap(pred_colors)
            
            ax[0][1].imshow(image_slice[1], cmap='gray', origin='lower')
            ax[0][1].imshow(combined_mask, cmap=cm, origin='lower')
            ax[0][1].set_title(title+dice)

            ax[1][0].imshow(image_slice[2], cmap='gray', origin='lower')
            ax[1][0].imshow(gt_mask, cmap=gm, origin='lower')
            ax[1][0].set_title("Ground Truth" + dice)

            ax[1][1].imshow(image_slice[3], cmap='gray', origin='lower')
            ax[1][1].imshow(pred_mask, cmap=pm, origin='lower')
            ax[1][1].set_title("Prediction" + dice)

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