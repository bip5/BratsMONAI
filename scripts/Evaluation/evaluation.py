import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')
import nibabel as nib

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
output_path,
slice_dice,
plot_output,
load_path,
plot_single,
eval_from_folder

)
from Input.dataset import (ExpDataset,ExpDatasetEval,
test_indices,train_indices,
val_indices,Brats23valDataset
)
from Input.localtransforms import test_transforms1,post_trans,train_transform,val_transform
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
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler

namespace = locals().copy()
config_dict=dict()
for name, value in namespace.items():
    if type(value) in [str,int,float,bool]:
        print(f"{name}: {value}")
        config_dict[f"{name}"]=value


dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
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
        
def plot_prediction(image_slice, gt_mask, pred_mask, o_path, title,plot_slice=plot_output):
    

    tp = len(np.nonzero(pred_mask & gt_mask)[0])
    fp = len(np.nonzero(pred_mask & ~gt_mask)[0])
    fn = len(np.nonzero(~pred_mask & gt_mask)[0])
    den = 2*tp + fp + fn
    if plot_slice:
        if den > 0:
            dice_value = 2 * tp / den
            if dice_value<0.1:
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

                ax[0][1].imshow(image_slice[1], cmap='gray', origin='lower')
                ax[0][1].imshow(combined_mask, cmap=cm, origin='lower')
                ax[0][1].set_title(title+dice)

                ax[1][0].imshow(image_slice[2], cmap='gray', origin='lower')
                ax[1][0].imshow(gt_mask, cmap='Greens', origin='lower')
                ax[1][0].set_title("Ground Truth" + dice)

                ax[1][1].imshow(image_slice[3], cmap='gray', origin='lower')
                ax[1][1].imshow(pred_mask, cmap='Reds', origin='lower')
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
        

def evaluate(eval_path,test_loader,output_path,model=model):
    print('function called')############################
    device = torch.device("cuda:0")
    model=torch.nn.DataParallel(model)
    model.to(device)
    
    # model.load_state_dict(torch.load(eval_path),strict=False)
    model.load_state_dict(torch.load(eval_path, map_location=lambda storage, loc: storage.cuda(0)), strict=False)
    
    model.eval()
    ind_scores = dict()
    # Initialize the Dice metric outside the loop
    slice_dice_metric = DiceMetric(include_background=True, reduction='none')
    slice_dice_scores = dict()
    slice_pred_area = dict()
    slice_gt_area = dict()
    with torch.no_grad():
        for test_data in test_loader: # each image
            
            test_inputs = test_data["image"].to(device) # pass to gpu
            test_labels=test_data["mask"].to(device)
            sub_id=test_data["id"][0]
            
            for idx, y in enumerate(test_labels):
                test_labels[idx] = (y > 0.5).int()
            
            test_outputs=inference(test_inputs,model)
            test_outputs=[post_trans(i) for i in decollate_batch(test_outputs)] 
                      
            dice_metric(y_pred=test_outputs, y=test_labels)
            dice_metric_batch(y_pred=test_outputs, y=test_labels)
            
            
            # Compute the Dice score for this test case
            current_dice = dice_metric.aggregate().item()
            ind_scores[sub_id]=round(current_dice,4)
            
            print('sub_id: ', sub_id)

            
            
            if slice_dice:
                # Select the 'whole tumor' class and drop the batch dimension
                whole_tumor = test_outputs[0][1].unsqueeze(0) # Adding batch dimension back to the start
                labels_tumor = test_labels[0][1].unsqueeze(0)
                
                

                # Transpose to treat each axial slice as a channel
                whole_tumor_transposed = whole_tumor.permute(0, 3, 1, 2)
                labels_tumor_transposed = labels_tumor.permute(0, 3, 1, 2)
                area_gt=[]
                area_pred=[]
                
                for i in range(whole_tumor_transposed.shape[1]):
                    pred_a=(whole_tumor_transposed[0][i]>0.5).sum()
                    gt_a=(labels_tumor_transposed[0][i]>0.5).sum()
                    area_pred.append(pred_a.cpu().numpy())
                    area_gt.append(gt_a.cpu().numpy())
                    
                # Now compute the Dice scores
                slice_dice_metric(y_pred=whole_tumor_transposed, y=labels_tumor_transposed)
                

                # Aggregate to get the slice-wise dice values
                slice_wise_dice_values = slice_dice_metric.aggregate().squeeze().cpu().numpy()
                
                slice_dice_metric.reset()
                slice_dice_scores[sub_id]=slice_wise_dice_values
                slice_gt_area[sub_id]=area_gt
                slice_pred_area[sub_id]=area_pred
                
                
            if plot_output:
                var_1=0
                if var_1==0:
                    os.makedirs(output_path,exist_ok=True)
                    cluster_fname=cluster_files.split('/')[-1]
                    
                    # Create a unique folder for each test case
                    case_save_path = os.path.join(output_path, f"id_{sub_id}_dice_{current_dice:.4f}")  # 4 decimal places for Dice score
                    print('MAKING FOLDER!')
                    os.makedirs(case_save_path, exist_ok=True)
                    #plot the slice with largest whole tumour for all tumour categories 
                    for mask_channel in range(test_labels.shape[1]):  # loop over the 3 mask channels
                        wt_channel=test_labels[0][1].cpu().numpy()
                        label_names=['TC','WT','ET']
                        gt_channel = test_labels[0][mask_channel].cpu().numpy()
                        pred_channel = test_outputs[0].cpu().numpy()[mask_channel]

                        centroid = find_centroid(gt_channel)
                        
                        for view, axis in enumerate(["Axial", "Sagittal", "Coronal"]):
                            # slice_idx = centroid[view]
                            axial_slices=test_data['image'][0].shape[3]
                            
                            # Axial view
                            if view == 0:
                                for slice_ in range(axial_slices):
                                    areas = wt_channel.sum(axis=(0, 1))  # Sum along the x and y axes
                                    slice_idx = areas.argmax()  # Get index of slice with max area
                                    
                                    if plot_single:
                                        if slice_==slice_idx:
                                            slice_data = test_data["image"][0][:, :, :, slice_idx].cpu().numpy()  # C=4, hence using channel 2
                                            gt_slice = gt_channel[:, :, slice_idx] > 0.5
                                            pred_slice = pred_channel[:, :, slice_idx] > 0.5
                                            title = f"{label_names[mask_channel]}_{axis}_s{slice_idx}_{sheet}"
                                            plot_prediction(slice_data, gt_slice, pred_slice, case_save_path, title) 
                                    else:
                                        slice_idx=slice_
                                        slice_data = test_data["image"][0][1, :, :, slice_idx].cpu().numpy()  # C=4, hence using channel 2
                                        gt_slice = gt_channel[:, :, slice_idx] > 0.5
                                        pred_slice = pred_channel[:, :, slice_idx] > 0.5
                                        title = f"{label_names[mask_channel]}_{axis}_s{slice_idx}_{sheet}"
                                        plot_prediction(slice_data, gt_slice, pred_slice, case_save_path, title) 
                                # elif view == 1:  # Sagittal view
                                    # pass
                                    # areas = wt_channel.sum(axis=(1, 2))  # Sum along the x and y axes
                                    # slice_idx = areas.argmax()  # Get index of slice with max area
                                    # slice_data = test_data["image"][0][1, slice_idx, :, :].cpu().numpy()
                                    # gt_slice = gt_channel[slice_idx, :, :] > 0.5
                                    # pred_slice = pred_channel[slice_idx, :, :] > 0.5
                                # else:  # Coronal view
                                    # pass
                                    # areas = wt_channel.sum(axis=(0, 2))  # Sum along the x and y axes
                                    # slice_idx = areas.argmax()  # Get index of slice with max area
                                    # slice_data = test_data["image"][0][1, :, slice_idx, :].cpu().numpy()
                                    # gt_slice = gt_channel[:, slice_idx, :] > 0.5
                                    # pred_slice = pred_channel[:, slice_idx, :] > 0.5

                            # title = f"{label_names[mask_channel]}_{axis}_s{slice_idx}_{cluster_fname[:6]}"
                            # plot_prediction(slice_data, gt_slice, pred_slice, case_save_path, title)  
                            # var_1+=1
                    # print('exiting now')
                    # sys.exit()
        metric_org = dice_metric.aggregate().item()
        
        metric_batch_org = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()

    metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()

    print("Metric on original image spacing: ", metric_org)
    print(f"metric_tc: {metric_tc:.4f}", f"   metric_wt: {metric_wt:.4f}", f"   metric_et: {metric_et:.4f}")
    return metric_org,metric_tc,metric_wt,metric_et,ind_scores,slice_dice_scores,slice_gt_area,slice_pred_area

if __name__ =='__main__':
    job_id = os.environ.get('SLURM_JOB_ID', 'N/A')
    if eval_mode=='cluster':
        test_count=30
        train_count=100
        
        print(len(np.unique(train_indices)),len(np.unique(test_indices)), len(np.unique(val_indices)))
     
        indices_dict=dict()
        indices_dict['Train indices']=train_indices
        indices_dict['Test indices']=test_indices
        indices_dict['Val indices']=val_indices
        print(indices_dict)
        
        xls = pd.ExcelFile(cluster_files) #to get sheets
        
       # Get all sheet names
        sheet_names = xls.sheet_names
        sheet_names=[x for x in sheet_names if 'Cluster_' in x]
        resultdict=dict()
        model_names=[]
        score_list=[]
        slice_dice_list=[]
        folder_name=eval_folder.split('/')[-1]
        # Get the current date and time
        now = datetime.datetime.now()
        if eval_from_folder:
            all_model_paths = [os.path.join(eval_folder, modelweights) for modelweights in os.listdir(eval_folder)]
        else:
            all_model_paths=[]
        all_model_paths.append(load_path)
        
        for path in sorted(all_model_paths):            
            eval_path = path
            modelweights=path.split('/')[-1]
            model_names.append(modelweights)
            dice_scores=[]
            for sheet in sheet_names:
                print('sheet' ,sheet) #################################
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
                    elif test_samples_from=='all':
                        test_sheet_i.append(i)
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
                test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)                
                
                

                # Convert it to a string in a specific format (e.g., YYYY-MM-DD_HH-MM-SS)
                formatted_time = modelweights+'_'+ sheet+ '_' +now.strftime('%Y-%m-%d_%H-%M-%S')     

                # Create the new directory path using the formatted time
                new_dir = os.path.join(output_path, formatted_time)

                # Make the new directory
                if slice_dice:
                    os.makedirs(new_dir, exist_ok=True)
                metric_org,metric_tc,metric_wt,metric_et,ind_scores,slice_dice_scores,slice_gt_area, slice_pred_area=evaluate(eval_path,test_loader,new_dir)
                # ind_scores['Cluster']=sheet
                # ind_scores['Model']=modelweights
                
                ind_score_df=pd.DataFrame(ind_scores.values(),index=ind_scores.keys(),columns=['Dice'])
                ind_score_df['Cluster']=sheet
                ind_score_df['Model']=modelweights
                score_list.append(ind_score_df)
                
                
                slice_dice_df=pd.DataFrame(slice_dice_scores)
                slice_gt_area_df = pd.DataFrame(slice_gt_area)
                slice_pred_area_df = pd.DataFrame(slice_pred_area)
                print(slice_dice_df.shape)
                print(slice_gt_area_df.shape)
                print(slice_pred_area_df.shape)

                excel_file_name = f'{new_dir}/sl_{sheet}_wm_{modelweights}_{folder_name}.xlsx'
                with pd.ExcelWriter(excel_file_name) as writer:
                    slice_dice_df.to_excel(writer, sheet_name="Dice Scores")
                    slice_gt_area_df.to_excel(writer, sheet_name="GT Area")
                    slice_pred_area_df.to_excel(writer, sheet_name="Predicted Area")
                print(f'saved in {new_dir}')
                for sample in range(slice_dice_df.shape[1]):
                    start=sample
                    end=sample+1
                    plot_scatter(slice_dice_df.iloc[:,start:end], slice_gt_area_df.iloc[:,start:end], slice_pred_area_df.iloc[:,start:end], f'{new_dir}/{sheet}_bubble{slice_dice_df.columns[sample]}.png')
                
                
                dice_scores.append(metric_org)
                config_dict['metric_org'] = metric_org
                config_dict['metric_tc'] = metric_tc
                config_dict['metric_wt'] = metric_wt
                config_dict['metric_et'] = metric_et
                config_dict['Cluster'] = evaluating_sheet
                log_path='/scratch/a.bip5/BraTS/eval_running_log.csv'
                log_run_details(config_dict,[],csv_file_path=log_path)
            resultdict['m'+ modelweights]=dice_scores
         
        ind_scores_df=pd.concat(score_list)
        evcl_name=cluster_files.split('/')[-1][:5]
        ind_scores_df.to_csv(f'./IndScores{folder_name}_{evcl_name}_{job_id}.csv')
          
        resultdict=pd.DataFrame(resultdict,index=sheet_names)
        resultdict.to_csv(f'./EvCl{folder_name}_{evcl_name}_{job_id}.csv')
        
    elif eval_mode=='online_val':
        
        val_path='/scratch/a.bip5/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
        test_ds=Brats23valDataset(val_path,transform=test_transforms1)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=workers)              
        device = torch.device("cuda:0")
        model=torch.nn.DataParallel(model)
        model.to(device)
        print(len(test_ds))
        # model.load_state_dict(torch.load(eval_path),strict=False)
        model.load_state_dict(torch.load(eval_path, map_location=lambda storage, loc: storage.cuda(0)), strict=False)
        with torch.no_grad():   

            for test_data in test_loader: # each image
                print('TRIGGERED')
                test_inputs = test_data["image"].to(device) # pass to gpu
                
                sub_id=test_data["id"][0]
                new_dir = os.path.join(output_path, 'brats23')
                os.makedirs(new_dir, exist_ok=True)
                
             
                test_outputs=inference(test_inputs,model)
                test_outputs=[post_trans(i) for i in decollate_batch(test_outputs)] 
                nii_img=nib.Nifti1Image(test_outputs[0].cpu().numpy(),np.eye(4))
                # Save the image to a .nii.gz file
                nii_img.to_filename(f'{new_dir}/{sub_id}.nii.gz')
                
    #print the script at the end of every run
    script_path = os.path.abspath(__file__) # Gets the absolute path of the current script
    with open(script_path, 'r') as script_file:
               script_content = script_file.read()
    print("\n\n------ Script Content ------\n")
    print(script_content)
    print("\n---------------------------\n")
                
    