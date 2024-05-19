import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')
import nibabel as nib


from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine
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
from Evaluation.eval_functions import inference,dice_metric_ind,dice_metric_ind_batch,dice_metric,dice_metric_batch,plot_expert_performance,distance_ensembler,model_loader
from Evaluation.eval_functions2 import evaluate_time_samples
import scipy.stats as stats
import matplotlib.pyplot as plt
print_config()

#trying to make the evaluation deterministic and independent to sample order
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config_path='/scratch/a.bip5/BraTS/scripts/Input/config.py'
with open(config_path, 'r') as config_file:
               config_content = config_file.read()
               
print("\n\n------ Config Content ------\n")
print(config_content)
print("\n---------------------------\n")


device = torch.device("cuda:0")
namespace = locals().copy()
config_dict=dict()
for name, value in namespace.items():
    if type(value) in [str,int,float,bool]:
        # print(f"{name}: {value}")
        config_dict[f"{name}"]=value


seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
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
        
   
        

def evaluate(eval_path,test_loader,output_path,model=model,**kwargs):
    print('function called')############################
    device = torch.device("cuda:0")
    
    model.to(device)
    
    plot_list=kwargs.get('plot_list',None)
    modelweights_folder_path=kwargs.get('modelweights_folder_path', None)
    # state_dict=torch.load(eval_path)
    ########Create a new state dict without the 'module.' prefix
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    
    # for k, v in state_dict.items():
        # name = k[7:]  # remove the 'module.' prefix
        # new_state_dict[name] = v
    ##### Load the modified state dict into the model
    # model.load_state_dict(new_state_dict)
    
    try:
        model.load_state_dict(torch.load(eval_path),strict=True)
        model=torch.nn.DataParallel(model)
    except:
        model=torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(eval_path),strict=True)
        
    # model.load_state_dict(torch.load(eval_path),strict=True)
    # model.load_state_dict(torch.load(eval_path, map_location=lambda storage, loc: storage.cuda(0)), strict=False)
    # model = torch.nn.DataParallel(model)
    model.eval()
    ind_scores = dict()
    # Initialize the Dice metric outside the loop
    slice_dice_metric = DiceMetric(include_background=True, reduction='none')
    slice_dice_scores = dict()
    slice_pred_area = dict()
    slice_gt_area = dict()
    # Save the current state of the weights before inference
    weights_before = {name: param.clone() for name, param in model.state_dict().items()}
    with torch.no_grad():
        for i,test_data in enumerate(test_loader): # each image
            
            test_inputs = test_data["image"].to(device) # pass to gpu
            sub_id = test_data["id"][0]
            if plot_list:
                if sub_id in plot_list:
                    pass
                else:
                    continue
            
            if eval_mode == 'cluster':
                print(sub_id, orig_i_list[i])
                # image_names=test_data[0]["imagepaths"]
            test_data["pred"] = inference(test_inputs,model)
            test_data=[post_trans(ii) for ii in decollate_batch(test_data)] #returns a list of n tensors where n=batch_size
            test_outputs,test_labels = from_engine(["pred","mask"])(test_data) # returns two lists of tensors
            
                        
            for idx, y in enumerate(test_labels):
                test_labels[idx] = (y > 0.5).int()
                
            test_outputs = [tensor.to(device) for tensor in test_outputs]
            test_labels = [tensor.to(device) for tensor in test_labels]

            # test_labels=test_data["mask"].to(device)
            sub_id=test_data[0]["id"]
            
            # torch.cuda.empty_cache()
            # print(type(test_outputs),test_labels[0].shape)
            
            dice_metric_ind(y_pred=test_outputs, y=test_labels)
            dice_metric_ind_batch(y_pred=test_outputs, y=test_labels)
            
            dice_metric(y_pred=test_outputs, y=test_labels)
            dice_metric_batch(y_pred=test_outputs, y=test_labels)
            
            
            # Compute the Dice score for this test case
            # current_dice = dice_metric_ind.aggregate().item()
            current_dice = dice_metric_ind.aggregate(reduction=None).item()
            batch_ind = dice_metric_ind_batch.aggregate()
            tc,wt,et = batch_ind[0].item(),batch_ind[1].item(),batch_ind[2].item()
            ind_scores[sub_id] = {'average':round(current_dice,4), 'tc':round(tc,4),'wt':round(wt,4),'et':round(et,4)}
            dice_metric_ind.reset()
            dice_metric_ind_batch.reset()
            
            # print('sub_id: ', sub_id)
            # print('image_names',image_names)

            
            
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
                
                os.makedirs(output_path,exist_ok=True)
                cluster_fname=cluster_files.split('/')[-1]
                
                # Create a unique folder for each test case
                case_save_path = os.path.join(output_path, f"id_{sub_id}_dice_{current_dice:.4f}")  # 4 decimal places for Dice score
                print('MAKING FOLDER!')
                os.makedirs(case_save_path, exist_ok=True)
                #plot the slice with largest whole tumour for all tumour categories 
                for mask_channel in range(test_labels[0].shape[0]):  # loop over the 3 mask channels
                    wt_channel=test_labels[0][1].cpu().numpy()
                    label_names=['TC','WT','ET']
                    gt_channel = test_labels[0][mask_channel].cpu().numpy()
                    pred_channel = test_outputs[0].cpu().numpy()[mask_channel]

                    centroid = find_centroid(gt_channel)
                    
                    for view, axis in enumerate(["Axial", "Sagittal", "Coronal"]):
                        # slice_idx = centroid[view]
                        axial_slices=test_data[0]['image'].shape[3]
                        
                        # Axial view
                        if view == 0:
                            for slice_ in range(axial_slices):
                                areas = wt_channel.sum(axis=(0, 1))  # Sum along the x and y axes
                                slice_idx = areas.argmax()  # Get index of slice with max area
                                
                                if plot_single_slice:
                                    if slice_==slice_idx:
                                        slice_data = test_data[0]["image"][:, :, :, slice_idx].cpu().numpy()  # C=4, hence using channel 2
                                        gt_slice = gt_channel[:, :, slice_idx] > 0.5
                                        pred_slice = pred_channel[:, :, slice_idx] > 0.5
                                        title = f"{label_names[mask_channel]}_{axis}_s{slice_idx}_{sheet}"
                                        plot_prediction(slice_data, gt_slice, pred_slice, case_save_path, title,plot_output) 
                                else:
                                    slice_idx=slice_
                                    slice_data = test_data[0]["image"][:, :, :, slice_idx].cpu().numpy()  # C=4, hence using channel 2
                                    gt_slice = gt_channel[:, :, slice_idx] > 0.5
                                    pred_slice = pred_channel[:, :, slice_idx] > 0.5
                                    title = f"{label_names[mask_channel]}_{axis}_s{slice_idx}_{sheet}"
                                    plot_prediction(slice_data, gt_slice, pred_slice, case_save_path, title,plot_output) 
                                
        metric_org = dice_metric.aggregate().item()
        
        metric_batch_org = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()
    
    # Check if weights have changed after inference
    weights_after = model.state_dict()
    for name, param in weights_after.items():
        assert torch.equal(weights_before[name], param), f"Weights changed for layer: {name}"
    metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()

    print("Metric on original image spacing: ", metric_org)
    print(f"metric_tc: {metric_tc:.4f}", f"   metric_wt: {metric_wt:.4f}", f"   metric_et: {metric_et:.4f}")
    return metric_org,metric_tc,metric_wt,metric_et,ind_scores,slice_dice_scores,slice_gt_area,slice_pred_area

if __name__ =='__main__':
    
    if model_name=='SegResNet_Flipper':
            
            val_transform=val_transform_Flipper
       
    job_id = os.environ.get('SLURM_JOB_ID', 'N/A')
    print(len(np.unique(train_indices)),len(np.unique(test_indices)), len(np.unique(val_indices)))
     
    indices_dict=dict()
    indices_dict['Train indices']=sorted(train_indices)
    indices_dict['Test indices']=sorted(test_indices)
    indices_dict['Val indices']=sorted(val_indices)
    print(indices_dict)
    
    xls = pd.ExcelFile(cluster_files) #to get sheets
    
    # Get all sheet names
    sheet_names = xls.sheet_names
    sheet_names=[x for x in sheet_names if 'Cluster_' in x]
    
    now = datetime.datetime.now()
    if eval_from_folder:
        all_model_paths = [os.path.join(eval_folder, modelweights) for modelweights in os.listdir(eval_folder) if modelweights.startswith('Cluster')]
    else:
        all_model_paths=[]
    all_model_paths.append(load_path)
    
    if eval_mode=='cluster_expert':

        print('CLUSTER EXPERT')
        model_names=[]
     
        folder_name=eval_folder.split('/')[-1]
        # Get the current date and time
        
     
        full_ds = BratsDataset(root_dir,transform=val_transform)
        test_ds = Subset(full_ds,train_indices)
        
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)                
        
        
        metric_org,metric_tc,metric_wt,metric_et,ind_scores,model_closest,dist_lists,slice_dice_scores,slice_gt_area, slice_pred_area = eval_model_selector(eval_folder,test_loader)#evaluate(eval_path,test_loader,new_dir,plot_list=plot_list)
        # ind_scores['Cluster']=sheet
        # ind_scores['Model']=modelweights
        
        ind_score_df=pd.DataFrame(ind_scores)
        ind_score_df=ind_score_df.T
        # print(model_closest)
        ind_score_df['Expert']=ind_score_df.index.map(model_closest)  
        
        dist_df = pd.DataFrame.from_dict(dist_lists, orient='index', columns=['d0', 'd1', 'd2', 'd3','dmin'])
        ind_score_df = ind_score_df.join(dist_df)
        print('print(ind_score_df.head)',ind_score_df.head())
        
        ind_score_df.reset_index(inplace=True)
        ind_score_df.rename(columns={'index': 'Subject ID'}, inplace=True)
       
        base_perf=pd.ExcelFile('/scratch/a.bip5/BraTS/IndScoresjob_7801281_7801402.xlsx' )#IndScoresm2023-12-07_18-50-58_cl4_m_7743367.xlsx')
        base_sheets=base_perf.sheet_names
        # for sheet in base_sheets:
            # base_df=base_perf.parse(sheet)            
            # base_df.drop(columns=['Original index', 'Cluster'],inplace=True)
            # base_df.rename(columns={'SegResNetCV1_j7688198ep128': f'Base {sheet}'},inplace=True)            
            # ind_score_df=ind_score_df.merge(base_df, on='Subject ID')         
            
        
   
        # # plot_expert_performance(ind_score_df,plots_dir)
        
        # t, p_val=stats.ttest_rel(ind_score_df['average'],ind_score_df['Base Average Dice'])
        # t_et, p_val_et=stats.ttest_rel(ind_score_df['et'],ind_score_df['Base ET'])
        # t_wt, p_val_wt=stats.ttest_rel(ind_score_df['wt'],ind_score_df['Base WT'])
        # t_tc, p_val_tc=stats.ttest_rel(ind_score_df['tc'],ind_score_df['Base TC'])
        # # base_perf_df=base_perf_df.rename(columns={'tc':'tc_base','wt':'wt_base','et': 'et_base'})
        
        
        # all_t = [t, t_et,t_wt,t_tc]
        # all_p = [p_val, p_val_et, p_val_wt, p_val_tc]
        # rows = ['Overall', 'et','wt','tc']
        # d = {'t-statistic':all_t,'p-value':all_p}
        # p_df = pd.DataFrame(d,index=rows)
        
        config_dict['metric_org'] = metric_org
        config_dict['metric_tc'] = metric_tc
        config_dict['metric_wt'] = metric_wt
        config_dict['metric_et'] = metric_et   
        config_dict['eval_job_id']=job_id

        log_path='/scratch/a.bip5/BraTS/eval_running_log.csv'
        log_run_details(config_dict,[],csv_file_path=log_path)
   
   
        writer=pd.ExcelWriter(f'./IndScores{folder_name}_{job_id}.xlsx',engine='xlsxwriter')
        ind_score_df.to_excel(writer,sheet_name='Everything',index=True)
        print('ind score df shape', ind_score_df.shape)
        # p_df.to_excel(writer,sheet_name='P_all')
        
        # for value in ind_score_df['Expert'].unique():
            # ind_score_filt=ind_score_df[ind_score_df['Expert']==value]
            # ind_score_filt.to_excel(writer,sheet_name=f'Expert {value}')            
            # t, p_val=stats.ttest_rel(ind_score_filt['average'],ind_score_filt['Base Average Dice'])
            # t_et, p_val_et=stats.ttest_rel(ind_score_filt['et'],ind_score_filt['Base ET'])
            # t_wt, p_val_wt=stats.ttest_rel(ind_score_filt['wt'],ind_score_filt['Base WT'])
            # t_tc, p_val_tc=stats.ttest_rel(ind_score_filt['tc'],ind_score_filt['Base TC'])
            # all_t = [t, t_et,t_wt,t_tc]
            # all_p = [p_val, p_val_et, p_val_wt, p_val_tc]
            # rows = ['Overall', 'et','wt','tc']
            # d = {'t-statistic':all_t,'p-value':all_p}
            # p_df = pd.DataFrame(d,index=rows)
            # p_df.to_excel(writer,sheet_name=f'P_{value}')
        
        modelname_df=pd.DataFrame(all_model_paths,columns=["Evaluated models"])
        modelname_df.to_excel(writer,sheet_name='RunInfo')
  
        writer.close()
        
    elif eval_mode=="distance_ensemble":
             # Get the current date and time
        print('DISTANCE ENSEMBLE')
        folder_name=eval_folder.split('/')[-1]
        full_ds = BratsDataset(root_dir,transform=val_transform)
        test_ds = Subset(full_ds,test_indices)
        
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)                
        
        
        metric_org,metric_tc,metric_wt,metric_et,ind_scores,model_closest,dist_lists,slice_dice_scores,slice_gt_area, slice_pred_area=distance_ensembler(eval_folder,test_loader)#evaluate(eval_path,test_loader,new_dir,plot_list=plot_list)
        # ind_scores['Cluster']=sheet
        # ind_scores['Model']=modelweights
        
        ind_score_df=pd.DataFrame(ind_scores)
        ind_score_df=ind_score_df.T
        # print(model_closest)
        ind_score_df['Expert']=ind_score_df.index.map(model_closest)  
        
        dist_df = pd.DataFrame.from_dict(dist_lists, orient='index', columns=['d0', 'd1', 'd2', 'd3','dmin'])
        ind_score_df = ind_score_df.join(dist_df)
        print('print(ind_score_df.head)',ind_score_df.head())
        
        ind_score_df.reset_index(inplace=True)
        ind_score_df.rename(columns={'index': 'Subject ID'}, inplace=True)
        base_perf=pd.ExcelFile('/scratch/a.bip5/BraTS/IndScoresm2023-12-07_18-50-58_cl4_m_7743367.xlsx')
        base_sheets=base_perf.sheet_names
        for sheet in base_sheets:
            base_df=base_perf.parse(sheet)
            base_df.drop(columns=['Original index', 'Cluster'],inplace=True)
            base_df.rename(columns={'SegResNetCV1_j7688198ep128': f'Base {sheet}'},inplace=True)
            ind_score_df=ind_score_df.merge(base_df, on='Subject ID')         
        
        
   
        plot_expert_performance(ind_score_df,plots_dir)
        
        t, p_val=stats.ttest_rel(ind_score_df['average'],ind_score_df['Base Average Dice'])
        t_et, p_val_et=stats.ttest_rel(ind_score_df['et'],ind_score_df['Base ET'])
        t_wt, p_val_wt=stats.ttest_rel(ind_score_df['wt'],ind_score_df['Base WT'])
        t_tc, p_val_tc=stats.ttest_rel(ind_score_df['tc'],ind_score_df['Base TC'])
        # base_perf_df=base_perf_df.rename(columns={'tc':'tc_base','wt':'wt_base','et': 'et_base'})
        
        
        all_t = [t, t_et,t_wt,t_tc]
        all_p = [p_val, p_val_et, p_val_wt, p_val_tc]
        rows = ['Overall', 'et','wt','tc']
        d = {'t-statistic':all_t,'p-value':all_p}
        p_df = pd.DataFrame(d,index=rows)
        
        config_dict['metric_org'] = metric_org
        config_dict['metric_tc'] = metric_tc
        config_dict['metric_wt'] = metric_wt
        config_dict['metric_et'] = metric_et   
        config_dict['eval_job_id']=job_id

        log_path='/scratch/a.bip5/BraTS/eval_running_log.csv'
        log_run_details(config_dict,[],csv_file_path=log_path)
   
   
        writer=pd.ExcelWriter(f'./IndEnsemble{folder_name}_{job_id}.xlsx',engine='xlsxwriter')
        ind_score_df.to_excel(writer,sheet_name='Everything',index=True)
        p_df.to_excel(writer,sheet_name='P_all')
        
        for value in ind_score_df['Expert'].unique():
            ind_score_filt=ind_score_df[ind_score_df['Expert']==value]
            ind_score_filt.to_excel(writer,sheet_name=f'Expert {value}')            
            t, p_val=stats.ttest_rel(ind_score_filt['average'],ind_score_filt['Base Average Dice'])
            t_et, p_val_et=stats.ttest_rel(ind_score_filt['et'],ind_score_filt['Base ET'])
            t_wt, p_val_wt=stats.ttest_rel(ind_score_filt['wt'],ind_score_filt['Base WT'])
            t_tc, p_val_tc=stats.ttest_rel(ind_score_filt['tc'],ind_score_filt['Base TC'])
            all_t = [t, t_et,t_wt,t_tc]
            all_p = [p_val, p_val_et, p_val_wt, p_val_tc]
            rows = ['Overall', 'et','wt','tc']
            d = {'t-statistic':all_t,'p-value':all_p}
            p_df = pd.DataFrame(d,index=rows)
            p_df.to_excel(writer,sheet_name=f'P_{value}')
        modelname_df=pd.DataFrame(all_model_paths,columns=["Evaluated models"])
        modelname_df.to_excel(writer,sheet_name='RunInfo')
        writer.close()
    
    elif eval_mode=='cluster':
        test_count=30
        train_count=100   
        
        
       
        resultdict=dict()
        model_names=[]
        score_list=[]
        slice_dice_list=[]
        folder_name=eval_folder.split('/')[-1]
    
        
        
        for path in sorted(all_model_paths):            
            eval_path = path
            modelweights=path.split('/')[-1]
            model_names.append(modelweights)
            dice_scores=[]
            for sheet in sheet_names:
                print('sheet' ,sheet) #################################
                test_sheet_i=[]
                orig_i_list=[]
                train_sheet_i=[]
                val_sheet_i=[]
                cluster_indices=pd.read_excel(cluster_files,sheet)['original index']
                
                for i,orig_i in enumerate(sorted(cluster_indices)):
                    
                    if test_samples_from=='trainval':
                        if orig_i in train_indices:
                            # while len(train_sheet_i)<train_count:
                            train_sheet_i.append(i)
                            orig_i_list.append(orig_i)
                        elif orig_i in val_indices:
                            val_sheet_i.append(i)
                            orig_i_list.append(orig_i)
                    elif test_samples_from=='val':
                        if orig_i in val_indices:
                            val_sheet_i.append(i)
                            orig_i_list.append(orig_i)
                    elif test_samples_from=='all':
                        test_sheet_i.append(i)
                        orig_i_list.append(orig_i)
                                        
                    else:                               
                        if orig_i in test_indices:
                           # print('orig_i ',orig_i)
                           test_sheet_i.append(i)
                           orig_i_list.append(orig_i)
                #will be adding empty list to give identical when mode not trainval
                test_sheet_i = test_sheet_i+train_sheet_i+val_sheet_i 
                
                if limit_samples:
                    test_sheet_i=test_sheet_i[:limit_samples]
                    orig_i_list=orig_i_list[:limit_samples]
                
                evaluating_sheet=sheet
                print(f'Evaluating {len(test_sheet_i)} samples from {evaluating_sheet} with saved model{eval_path}')
                
                test_ds = ExpDataset(cluster_files,evaluating_sheet,transform=val_transform)#creating dataset from sheet in xls
                test_ds = Subset(test_ds,test_sheet_i)
                
                test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)                
                
                

                # Convert it to a string in a specific format (e.g., YYYY-MM-DD_HH-MM-SS)
                formatted_time = modelweights+'_'+ sheet+ '_' +now.strftime('%Y-%m-%d_%H-%M-%S')+str(plot_output)     

                # Create the new directory path using the formatted time
                new_dir = os.path.join(output_path, formatted_time)

                # Make the new directory
                if slice_dice:
                    os.makedirs(new_dir, exist_ok=True)
                metric_org,metric_tc,metric_wt,metric_et,ind_scores,slice_dice_scores,slice_gt_area, slice_pred_area=evaluate(eval_path,test_loader,new_dir,plot_list=plot_list)##eval_model_selector(eval_folder,test_loader,model)#
                # ind_scores['Cluster']=sheet
                # ind_scores['Model']=modelweights
                
                ind_score_df=pd.DataFrame(ind_scores)
                ind_score_df=ind_score_df.T
                ind_score_df['Cluster']=sheet
                ind_score_df['Model']=modelweights
                ind_score_df['Original index']= orig_i_list
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
                config_dict['eval_job_id']=job_id

                log_path='/scratch/a.bip5/BraTS/eval_running_log.csv'
                log_run_details(config_dict,[],csv_file_path=log_path)
            resultdict['m'+ modelweights]=dice_scores
         
        ind_scores_df=pd.concat(score_list)
        evcl_name=cluster_files.split('/')[-1][:5]
        print(ind_scores_df.columns)
        print(ind_scores_df.head())
        ind_scores_df.reset_index(inplace=True)
        ind_scores_df.rename(columns={'index': 'Subject ID'}, inplace=True)
        
        

        writer=pd.ExcelWriter(f'./IndScores{folder_name}_{evcl_name}_{job_id}.xlsx',engine='xlsxwriter')
        ind_scores_piv=ind_scores_df.pivot_table(
        index=['Subject ID', 'Original index','Cluster'],
        columns='Model',
        values='average',
        aggfunc='first'
        ).reset_index()
        
        ind_scores_piv.to_excel(writer,sheet_name='Average Dice',index=False)
        
        ind_scores_piv=ind_scores_df.pivot_table(
        index=['Subject ID', 'Original index','Cluster'],
        columns='Model',
        values='wt',
        aggfunc='first'
        ).reset_index()
        
        ind_scores_piv.to_excel(writer,sheet_name='WT',index=False)
        
        ind_scores_piv=ind_scores_df.pivot_table(
        index=['Subject ID', 'Original index','Cluster'],
        columns='Model',
        values='tc',
        aggfunc='first'
        ).reset_index()
        
        ind_scores_piv.to_excel(writer,sheet_name='TC',index=False)
        
        ind_scores_piv=ind_scores_df.pivot_table(
        index=['Subject ID', 'Original index','Cluster'],
        columns='Model',
        values='et',
        aggfunc='first'
        ).reset_index()
        
        ind_scores_piv.to_excel(writer,sheet_name='ET',index=False)
        
        modelname_df=pd.DataFrame(all_model_paths,columns=["Evaluated models"])
        modelname_df.to_excel(writer,sheet_name='RunInfo')
          
        resultdict=pd.DataFrame(resultdict,index=sheet_names)
        print(resultdict.shape,'resultdict.shape')
        resultdict.to_csv(f'./EvCl{folder_name}_{evcl_name}_{job_id}.csv')
        writer.save()
    elif eval_mode=='online_val':
        non_val=np.concatenate([test_indices,val_indices,train_indices])
        
        model_names=[]
        score_list=[]
        slice_dice_list=[]
        folder_name=eval_folder.split('/')[-1]
        # Get the current date and time
        
        if use_cluster_for_online_val:
            for path in sorted(all_model_paths[:-1]):            
                eval_path = path
                modelweights=path.split('/')[-1]
                model_names.append(modelweights)
                dice_scores=[]
                for sheet in sheet_names:
                    print(modelweights[:9], sheet)
                    if modelweights[:9]==sheet:
                         #################################
                        test_sheet_i=[]
                        orig_i_list=[]
                        
                        cluster_indices=pd.read_excel(cluster_files,sheet)['original index']
                        for i,orig_i in enumerate(sorted(cluster_indices)):
                            if orig_i not in non_val:
                                print(orig_i, 'orig_i')
                                test_sheet_i.append(i)
                                orig_i_list.append(orig_i)
                        print(test_sheet_i,'test_sheet_i')       
                        print(f'Generating {len(test_sheet_i)} samples from {sheet} with saved model{eval_path}')
                        
                        test_ds = ExpDatasetEval(cluster_files,sheet,transform=test_transforms1)
                        test_ds = Subset(test_ds,test_sheet_i)
                        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4) 
                        device = torch.device("cuda:0")
                        model=torch.nn.DataParallel(model)
                        model.to(device)
                        model.load_state_dict(torch.load(eval_path),strict=True)
                        with torch.no_grad():   

                            for test_data in test_loader: # each image
                                
                                test_inputs = test_data["image"].to(device) # pass to gpu
                                
                                sub_id=test_data["id"][0]
                                print(sub_id)
                                try:
                                    test_data["pred"]=inference(test_inputs,model)
                                    test_data=[post_trans_test(i) for i in decollate_batch(test_data)]
                                    test_outputs=from_engine(["pred"])(test_data) 
                                    
                                    nampendix=eval_path.split('/')[-1]
                                    # returns a list of tensors
                                    new_dir = os.path.join(output_path, f'brats23_{nampendix}')
                                    os.makedirs(new_dir, exist_ok=True)
                                               
                                                    
                                    print(test_outputs[0].shape)
                                    nii_img=nib.Nifti1Image(test_outputs[0].numpy(),np.eye(4))
                                    # Save the image to a .nii.gz file
                                    nii_img.to_filename(f'{new_dir}/{sub_id}.nii.gz')
                                except:
                                    print(sub_id, 'issue with this file')                
                                    continue
        else:
            val_path='/scratch/a.bip5/BraTS/GLIValidationData'
            test_ds = Brats23valDataset(val_path,transform=test_transforms1)
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)              
            device = torch.device("cuda:0")
            model=torch.nn.DataParallel(model)
            model.to(device)
            print(len(test_ds))
            model.load_state_dict(torch.load(load_path),strict=False)
            ##model.load_state_dict(torch.load(eval_path, map_location=lambda storage, loc: storage.cuda(0)), strict=False)
            with torch.no_grad():   

                for test_data in test_loader: # each image
                    print('TRIGGERED')
                    test_inputs = test_data["image"].to(device) # pass to gpu
                    
                    sub_id=test_data["id"][0]
                    print(sub_id)
                    try:
                        test_data["pred"]=inference(test_inputs,model)
                        test_data=[post_trans_test(i) for i in decollate_batch(test_data)]
                        test_outputs=from_engine(["pred"])(test_data) 
                        
                        nampendix=eval_path.split('/')[-1]
                        #returns a list of tensors
                        new_dir = os.path.join(output_path, f'brats23_{nampendix}')
                        os.makedirs(new_dir, exist_ok=True)
                                   
                                        
                        print(test_outputs[0].shape)
                        nii_img=nib.Nifti1Image(test_outputs[0].numpy(),np.eye(4))
                        ##Save the image to a .nii.gz file
                        nii_img.to_filename(f'{new_dir}/{sub_id}.nii.gz')
                    except:
                        print(sub_id, 'issue with this file')                
                        continue
    
    elif eval_mode=='time':
        old_ds = BratsTimeDataset(root_dir,0,transform=val_transform)
        new_ds = BratsTimeDataset(root_dir,1,transform=val_transform)
        ## here we're creating two loaders to be able to call a mask from a different dataset
        old_loader=DataLoader(old_ds,shuffle=False,batch_size=1,num_workers=4)
        new_loader=DataLoader(new_ds,shuffle=False,batch_size=1,num_workers=4)
        out=evaluate_time_samples(load_path,old_loader,new_loader,eval_folder,expert=True,ensemble=True)
        
        
        
        ind_scores,ind_scores_new,ind_scores_newold,ind_scores_oldnew,ind_scores_GTGT=[pd.DataFrame(df) for df in out]
        
        ind_scores=ind_scores.T
        ind_scores_new=ind_scores_new.T
        ind_scores_newold=ind_scores_newold.T
        ind_scores_oldnew=ind_scores_oldnew.T
        ind_scores_GTGT = ind_scores_GTGT.T
        
        ind_scores.reset_index(inplace=True)
        

        
        ind_scores.rename(columns={x: x+'_old' for x in ind_scores.columns if x not in ['index', 'new in', 'old in','predicted volume delta', 'old volume gt','new volume gt','GT delta','old volume pred','new volume pred','pred delta','marker_color','d average','d tc', 'd wt', 'd et']}, inplace=True)
        print(ind_scores.columns)
        print(ind_scores.head())
        
        ind_scores_new.rename(columns={x: x+'_new' for x in ind_scores_new.columns if x != 'index'}, inplace=True)

        ind_scores_newold.rename(columns={x: x+'_newSoldT' for x in ind_scores_newold.columns if x not in ['index', 'new in', 'old in']}, inplace=True)
        ind_scores_oldnew.rename(columns={x: x+'_oldSnewT' for x in ind_scores_oldnew.columns if x != 'index'}, inplace=True)
        ind_scores_GTGT.rename(columns={x: x+'_GTGT' for x in ind_scores_GTGT.columns if x != 'index'}, inplace=True)
        
        final_df=pd.merge(ind_scores,ind_scores_new, on='index', how='inner')
        final_df=pd.merge(final_df,ind_scores_newold, on='index' ,how='inner')
        final_df=pd.merge(final_df,ind_scores_oldnew, on='index', how='inner')
        final_df=pd.merge(final_df,ind_scores_GTGT, on='index', how='inner')
        final_df.to_csv(f'./expert_time_samp_{job_id}.csv')
        save_path= os.path.join(plots_dir,f'expert_time_Dice_avg_plot_{job_id}.png')
        plt.figure()
        plt.scatter(final_df['average_GTGT'],final_df['d average'],c=final_df['marker_color'], alpha=0.5)
        plt.xlabel('Dice score between the two Ground Truths')
        plt.ylabel('Dice score for delta volume expert model segmentation')
        plt.ylim(0.4,1)
        plt.xlim(0,1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        
        plt.close()
        
    elif eval_mode=='simple':
        
        modelname = load_path.split('/')[-1]
        full_dataset = BratsDataset(root_dir,transform=val_transform)
        test_indices=test_indices.tolist()
        test_dataset = Subset(full_dataset, test_indices)
        test_loader=DataLoader(test_dataset,shuffle=False,batch_size=1,num_workers=4)
        
        # Convert it to a string in a specific format (e.g., YYYY-MM-DD_HH-MM-SS)
        formatted_time = modelname + now.strftime('%Y-%m-%d_%H-%M-%S')    

        # Create the new directory path using the formatted time
        new_dir = os.path.join(output_path, formatted_time)
        
        metric_org,metric_tc,metric_wt,metric_et,ind_scores,slice_dice_scores,slice_gt_area, slice_pred_area=evaluate(load_path,test_loader,new_dir)
        
        print("Metric on original image spacing: ", metric_org)
        print(f"metric_tc: {metric_tc:.4f}", f"   metric_wt: {metric_wt:.4f}", f"   metric_et: {metric_et:.4f}")
        
                
    #print the script at the end of every run
    script_path = os.path.abspath(__file__) # Gets the absolute path of the current script
    with open(script_path, 'r') as script_file:
               script_content = script_file.read()
    print("\n\n------ Script Content ------\n")
    print(script_content)
    print("\n---------------------------\n")
                
    