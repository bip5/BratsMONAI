from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd
# print(pandas.__version__)
import nibabel as nb


import os
import monai
from monai.data import Dataset
from monai.utils import set_determinism
from sisa import SISANet
import matplotlib.pyplot as plt
import cv2

from monai.transforms import (EnsureChannelFirstD, AddChannelD,\
    ScaleIntensityD, SpacingD, OrientationD,\
    ResizeD, RandAffineD,
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    RandBiasFieldD,
    RandRotateD,
    RotateD, Rotate,
    RandGaussianSmoothD,
    GaussianSmoothD,
    RandGaussianNoised,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd, RandFlip,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,   
    ToTensorD,
    EnsureTyped,
    AdjustContrastD,
    RandKSpaceSpikeNoiseD,
    RandGaussianSharpenD,
    SaveImage,SaveImaged,

    MeanEnsembled,
    VoteEnsembled,
    EnsureType,
    Activations,
    SplitChanneld,
)



from monai.engines import (
    EnsembleEvaluator,
    SupervisedEvaluator,
    SupervisedTrainer
)
from monai.handlers.ignite_metric import IgniteMetric
from monai.config.type_definitions import NdarrayOrTensor
from monai.networks import one_hot
from monai.networks.layers import GaussianFilter, apply_filter

from monai.transforms.utils import fill_holes, get_largest_connected_component_mask
from monai.metrics.utils import do_metric_reduction
from monai.transforms.utils_pytorch_numpy_unification import unravel_index
from monai.utils import convert_data_type, deprecated_arg, ensure_tuple, look_up_option
from monai.utils.type_conversion import convert_to_dst_type
from monai.losses import DiceLoss
from monai.utils import UpsampleMode,MetricReduction
from monai.data import decollate_batch, list_data_collate
from monai.handlers import MeanDice, StatsHandler, ValidationHandler, from_engine,HausdorffDistance
from monai.networks.nets import SegResNet,UNet
from monai.metrics import DiceMetric,compute_meandice,compute_hausdorff_distance,IterationMetric,CumulativeIterationMetric
from monai.inferers import sliding_window_inference
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.data import DataLoader
import numpy as np
from datetime import date, datetime
import sys
import re
import torch
import time

from torch.utils.data import Subset
import argparse
from torchsummary import summary
import gc
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from eval_utils import *
from Input.dataset import test_indices,ExpDatasetEval
from eval_config import *
from brats_transforms import *
from Evaluation.evaluation import (
inference,

)

from Training.network import model
from Input.localtransforms import (
train_transform,
val_transform,
post_trans,
)
from Input.config import load_path
import eval_config
import warnings

test_indices=np.arange(150,200)

if __name__=="__main__":
    parser=argparse.ArgumentParser(prog=sys.argv[0],description="Eval parser")

    parser.add_argument("--load_barlow",default =0, type=int,help="flag to use barlow twins backbone to initialise weight")
    parser.add_argument("--barlow_final",default=1, type=int, help="flag to use checkpoint instead of final model for barlow")
    parser.add_argument("--bar_model_name",default="checkpoint.pth", type=str,help="model name to load")

    args=parser.parse_args()

    print(' '.join(sys.argv))


    set_determinism(seed=0)


    # Get a dictionary of the current global namespace
    namespace = globals().copy()

    for name, value in namespace.items():
        print(f"{name}: {value}")

    warnings.filterwarnings("ignore")
    ##################~~~~~~~~~~~~~~~~~~~~~~~~~Model Definition~~~~~~~~~~~~~~~~~~~~~~~#################
 

    # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    # if model=="UNet":
         # model=UNet(
            # spatial_dims=3,
            # in_channels=4,
            # out_channels=3,
            # channels=(64,128,256,512,1024),
            # strides=(2,2,2,2)
            # ).to(device)
    # elif model=="SegResNet":
        # model = SegResNet(
            # blocks_down=[1, 2, 2, 4],
            # blocks_up=[1, 1, 1],
            # init_filters=32,
            # norm="instance",
            # in_channels=4,
            # out_channels=3,
            # upsample_mode=UpsampleMode[upsample]    
            # ).to(device)

    # else:
        # model = locals() [model](4,3).to(device)

    with torch.cuda.amp.autocast():
        summary(model,(4,96,96,72)) #just double all sizes
    model=torch.nn.DataParallel(model)
   
    print("Model defined and passed to GPU")

    loss_function = loss_function
    optimizer=getattr(torch.optim,optimizer_name)
    optimizer=optimizer(model.parameters(), lr, weight_decay=weight_decay) 
    scheduler=getattr(torch.optim.lr_scheduler,lr_scheduler_name)
    lr_scheduler = scheduler(optimizer, T_max=schedule_epochs)
  
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    post_trans = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    )
    if args.load_barlow==1:
        # Load model weight from Barlow Twins
        ckpt=torch.load("./barlowtwins/checkpoint/"+args.bar_model_name)
        if args.barlow_final==0:
            # print(ckpt.keys())
            for k in model.state_dict().keys():
                # print(ckpt["model"].keys())
                ckpt["model"][k]=ckpt["model"].pop("module.backbone."+k[7:])           
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            model.load_state_dict(ckpt,strict=False)

        print("Model weights loaded from pretrained Barlow Twins Backbone")

    if load_save==1:    
        ckpt=torch.load(load_path)        
        model.load_state_dict(ckpt, strict=False)
        print("Model weights loaded from best metric model")
        
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

           
    if val==10: 
        all1=make_dataset("/scratch/a.bip5/BraTS/BraTS_23_training")
        
        gt_used=all1[1]#[:10]
        imageall=all1[0]#[:10]
        # imagef=[i[0] for i in imageall]
        
    elif val==1:
        val_indices=np.random.choice(np.arange(1000),max_samples,replace=False)
        test_ds=TestDataset("/scratch/a.bip5/BraTS/BraTS_23_training",transform=test_transforms0)
        test_ds=Subset(test_ds,val_indices)
        gt_all=make_dataset("/scratch/a.bip5/BraTS/BraTS_23_training")[1]
        gt_used=[gt_all[i] for i in val_indices]
        
        #print("list of input gt files",make_dataset("./RSNA_ASNR_MICCAI_BraTS2021_TestData")[1])
    else:
        print('Testing on test set data')
        # val_indices=np.random.choice(np.arange(200),max_samples,replace=False)
        # test_ds=TestDataset(data_dir,transform=test_transforms0)
        test_indices=np.arange(150,200)
        # test_ds=Subset(test_ds,test_indices)
        files='/scratch/a.bip5/BraTS 2021/selected_files_seed1693762864.xlsx'
    
        xls = pd.ExcelFile(files)
        # Get all sheet names
        sheet_names = xls.sheet_names
        evaluating_sheet=sheet_names[3]
        print(f'Evaluating {len(test_indices)} samples from {evaluating_sheet} with the path{eval_path}')
        
        test_ds=ExpDatasetEval(files,evaluating_sheet,transform=test_transforms0)
        test_ds=Subset(test_ds,test_indices)
        # gt_all=make_dataset('/scratch/a.bip5/BraTS 2021/BraTS21_data')[1]
        # gt_used=[gt_all[i] for i in test_indices]
        
    
        
      
        
    if val==10:
        features=mask_feat(imageall,gt_used) 
        
        # features=features.assign(**{
            # 'path':gt_used,
            # 'tumour_core': tumour_core,
            # 'enhancing tumor': ent,
            # 'edema': ed,
            # 'size': size,
            # 'sagittal_profile_tc': sagittal_profile_tc,
            # 'frontal_profile_tc': frontal_profile_tc,
            # 'axial_profile_tc': axial_profile_tc,
            # 'sagittal_profile_wt': sagittal_profile_wt,
            # 'frontal_profile_wt': frontal_profile_wt,
            # 'axial_profile_wt': axial_profile_wt,
            # 'sagittal_profile_et': sagittal_profile_et,
            # 'frontal_profile_et': frontal_profile_et,
            # 'axial_profile_et': axial_profile_et,
            # 'sagittal_reg': sagittal_reg,
            # 'frontal_reg': frontal_reg,
            # 'axial_reg': axial_reg,
            # 'sagittal_da_profile': sagittal_da_profile,
            # 'frontal_da_profile': frontal_da_profile,
            # 'axial_da_profile': axial_da_profile,
            # 'reg_score_avg': reg_score_avg,
            # 'da_prof_avg': da_prof_avg,
            # 'contrast_a': contrast_a, 
            # 'dissimilarity_a': dissimilarity_a, 
            # 'homogeneity_a': homogeneity_a, 
            # 'energy_a': energy_a, 
            # 'correlation_a': correlation_a,
            # 's_contrast': s_contrast, 
            # 's_dissimilarity': s_dissimilarity, 
            # 's_homogeneity': s_homogeneity, 
            # 's_energy': s_energy, 
            # 's_correlation': s_correlation,
            # 'c_contrast': c_contrast, 
            # 'c_dissimilarity': c_dissimilarity, 
            # 'c_homogeneity': c_homogeneity, 
            # 'c_energy': c_energy, 
            # 'c_correlation': c_correlation,

            # })
        features.to_csv(f'Dataset_features{round(time.time())}.csv')
        sys.exit()

        
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers) # this should return 10 different instances

    # print("input type",type(next(iter(test_loader))))





    post_transforms = Compose([
        EnsureTyped(keys=["pred","label"]), 
        Activationsd(keys="pred", sigmoid=True),    
        AsDiscreted(keys="pred", threshold=0.5),
        # SaveImaged(keys=["pred","label"],meta_keys="pred_meta_dict",output_dir="./ssensemblemodels0922/outputs/",resample=False)
    ])
    post_pred= Compose([EnsureType(),Activations(sigmoid=True), AsDiscrete(threshold=0.2)])
    post_label = Compose([EnsureType()])
    
    print(type(model))

    if ensemble==1:
    
        if eval_config.model=="UNet":
            model_names= os.listdir(eval_path)
            #["UNetep100rs4C","UNetep99rs4C","UNetep98rs4C","UNetep97rs4C","UNetep96rs4C"]


            print(model_names)
            # wts=[0.69,0.69,0.78,0.72,0.62,0.7,0.7,0.7,0.75,0.67]
       
        elif eval_config.model=="SegResNet":
            model_names=os.listdir(eval_path)#'/scratch/a.bip5/BraTS 2021/ssensemblemodels0922/Evaluation Folder1')#["SegResNetep100rs2C","SegResNetep99rs2C","SegResNetep98rs2C","SegResNetep97rs2C","SegResNetep96rs2C"]          
            print(model_names)
            # wts=[0.7401,0.7506,0.6364,0.7484, 0.6725,0.7707,0.7219,0.7439,0.8003,0.7458]#[0.5651,0.5252,0.5537,0.5137,0.5744,0.4862,0.5255,0.5559,0.5755,0.5060]
        

        else:
            print("No SISA yet")
        

        
        
        model_num= len(model_names)
        mean_dice=[]
        hausdorff=[]
        tumor_core=[]
        whole_tumor=[]
        enhancing_tumor=[]
        scores={}
        scores["GT path"]=gt_used
    
        
        if plot>0:
            for i in range(int(model_num)): #//5)): #cycle through to each of the model name 
                if plot==1:
                    model_steps1=[model_names[i]]
                    # if i%5!=0:
                     # continue
                    model_steps=model_names#[:i+1]#[:(i+1)]
                    print(model_steps)  
                elif plot==2:
                    model_steps1=[model_names[i]]
                    model_steps=[model_names[i]]
                    print(model_steps)                   
                    
                models=[]
                models1=[]
                for name in model_steps:
                   
                    if eval_config.model=="UNet":
                         model=UNet(
                            spatial_dims=3,
                            in_channels=4,
                            out_channels=3,
                            channels=(64,128,256,512,1024),
                            strides=(2,2,2,2)
                            ).to(device)
                    elif eval_config.model=="SegResNet":
                        model = SegResNet(
                            blocks_down=[1, 2, 2, 4],
                            blocks_up=[1, 1, 1],
                            init_filters=32,
                            norm="instance",
                            in_channels=4,
                            out_channels=3,
                            upsample_mode=UpsampleMode[upsample]    
                            ).to(device)

                    else:
                        model = locals() [model](4,3).to(device)
                    
                    model=torch.nn.DataParallel(model)
                    
                    model.load_state_dict(torch.load(eval_path+'/'+name),strict=False)
                    model.eval()
                    models.append(model)
                    
                for name in model_steps1:
               
                    if eval_config.model=="UNet":
                         model1=UNet(
                            spatial_dims=3,
                            in_channels=4,
                            out_channels=3,
                            channels=(64,128,256,512,1024),
                            strides=(2,2,2,2)
                            ).to(device)
                    elif eval_config.model=="SegResNet":
                        model1 = SegResNet(
                            blocks_down=[1, 2, 2, 4],
                            blocks_up=[1, 1, 1],
                            init_filters=32,
                            norm="instance",
                            in_channels=4,
                            out_channels=3,
                            upsample_mode=UpsampleMode[upsample]    
                            ).to(device)

                    else:
                        model1 = locals() [model1](4,3).to(device)
                    
                    model1=torch.nn.DataParallel(model1)
                    
                    model1.load_state_dict(torch.load(eval_path+'/'+name),strict=False)
                    model1.eval()
                    models1.append(model1)
                        
                num_models=len(models)  
                if avgmodel:
                    for key in model.state_dict().keys():
                        for i in models[:-1]:
                            model.state_dict()[key]=(i.state_dict()[key]+model.state_dict()[key] )
                        model.state_dict()[key]=model.state_dict()[key]/num_models
                    torch.save(
                        model.state_dict(),
                        os.path.join("./saved models", date.today().isoformat()+'T'+str(datetime.today().hour)+ model+"zoo_avg5160"))
                        
                    print("saved zoo model")
                    
                    ensemble = 0
                    load_name=date.today().isoformat()+'T'+str(datetime.today().hour)+ model+"zoo_avg"
                    
                    break
                        
                   
                                
      
                
                mean_post_transforms = Compose(
                    [
                        EnsureTyped(keys=["pred"+str(i) for i in range(len(models))]), #gives pred0..pred1...
                        ## SplitChanneld(keys=["pred"+str(i) for i in range(10)]),
                        
                        MeanEnsembled(
                            keys=["pred"+str(i) for i in range(len(models))], 
                            output_key="pred",
                          ##  # in this particular example, we use validation metrics as weights
                          ### weights=wts,
                        ),
                        Activationsd(keys="pred", sigmoid=True),
                        AsDiscreted(keys="pred", threshold=0.2),
                        SaveImaged(keys="pred",output_dir='/scratch/a.bip5/BraTS 2021/ssensemblemodels0922/outputs', output_ext=".nii.gz",meta_key_postfix=None, output_postfix= str(val),print_log=False)
                    ]
                ) 
                
                mean_post_transforms1 = Compose(
                    [
                        EnsureTyped(keys=["pred"+str(i) for i in range(len(models1))]), #gives pred0..pred1...
                        ## SplitChanneld(keys=["pred"+str(i) for i in range(10)]),
                       
                                        
                        MeanEnsembled(
                            keys=["pred"+str(i) for i in range(len(models1))], 
                            output_key="pred",
                          ##  # in this particular example, we use validation metrics as weights
                          ### weights=wts,
                        ),
                        Activationsd(keys=["pred"+str(i) for i in range(len(models1))], sigmoid=True),
                        AsDiscreted(keys=["pred"+str(i) for i in range(len(models1))], threshold=0.5),
                    ]
                )
                                   
                
                
                md1,tc1,wt1,et1,haus1,pred_size1,_,_,_,_,_,_,_,_,_,_,_=ensemble_evaluate(mean_post_transforms1, models1,test_loader)
                scores[model_steps1[0]]=md1
                scores['haus_'+ model_steps1[0]]=haus1
                if i==4:
                    md,tc,wt,et,haus,pred_size,pred_tc,pred_et,pred_dp_sag_tc,pred_dp_sag_wt,pred_dp_sag_et,pred_dp_fr_tc,pred_dp_fr_wt,pred_dp_fr_et,pred_dp_ax_tc,pred_dp_ax_wt,pred_dp_ax_et=ensemble_evaluate(mean_post_transforms, models)
                    #if val==1: # in case we want individualised score only for train data
                    scores["Ensemble"]=md
                    scores["HausEnsemble"]=haus
                if plot==1:
                    mean_dice.append(md1.tolist())
                    hausdorff.append(haus1.tolist())
                    tumor_core.append(tc1)
                    whole_tumor.append(wt1)
                    enhancing_tumor.append(et1)
                else:
                    mean_dice.append(md1)
                    hausdorff.append(haus1)
                    tumor_core.append(tc1)
                    whole_tumor.append(wt1)
                    enhancing_tumor.append(et1)
                del models
                gc.collect()
                torch.cuda.empty_cache()
            # if val==1:
                    # sorted_scores=dict(sorted(scores.items(),key=lambda item: item[1])) # sorts the models by score
                    # print (sorted_scores)
            # print(mean_dice,'mean_dice')
            mean_dice_best=np.array(mean_dice).max(axis=0)
            mean_dice_model=np.array(mean_dice).max(axis=1)
            actual_mean_dice=np.array(mean_dice).mean()
            print('actual_mean_dice',actual_mean_dice)
            print("the best average mean dice from best results is", mean_dice_best.mean())
            scores_df=pd.DataFrame(scores)
            scores_df["Best perf"]=scores_df.iloc[:,1:11:2].max(axis=1)
            scores_df["Best model"]=scores_df.iloc[:,1:11:2].to_numpy().argmax(axis=1)
            scores_df["Tumour size"]=size           
            scores_df["predWT"]=pred_size
            scores_df["Enhancing Tumour"]=ent
            scores_df['predET']=pred_et
            scores_df["Tumour core"]=tumour_core
            scores_df['predTC']=pred_tc
            scores_df["edema"]=ed
            scores_df['sagittal_profile_tc']=sagittal_profile_tc
            scores_df['pred_dp_sag_tc']=pred_dp_sag_tc
            scores_df['sagittal_profile_wt']=sagittal_profile_wt
            scores_df['pred_dp_sag_wt']=pred_dp_sag_wt
            scores_df['sagittal_profile_et']=sagittal_profile_et
            scores_df['pred_dp_sag_et']=pred_dp_sag_et
            scores_df['frontal_profile_tc']=frontal_profile_tc
            scores_df['pred_dp_fr_tc']=pred_dp_fr_tc
            scores_df['frontal_profile_wt']=frontal_profile_wt
            scores_df['pred_dp_fr_wt']=pred_dp_fr_wt
            scores_df['frontal_profile_et']=frontal_profile_et
            scores_df['pred_dp_fr_et']=pred_dp_fr_et
            scores_df['axial_profile_tc']=axial_profile_tc
            scores_df['pred_dp_ax_tc']=pred_dp_ax_tc
            scores_df['axial_profile_wt']=axial_profile_wt
            scores_df['pred_dp_ax_wt']=pred_dp_ax_wt
            scores_df['axial_profile_et']=axial_profile_et
            scores_df['pred_dp_ax_et']=pred_dp_ax_et          
       
            # scores_df["Dice Prof Avg"]=avg_dice_profile
            scores_df["Regularity avg"]=reg_score_avg
            scores_df["Delta avg"]=da_prof_avg
            scores_df["ratio et/total"]=np.array(ent)/np.array(size)
            scores_df["tumour_core/total"]=np.array(tumour_core)/np.array(size)
            scores_df["ed/total"]=np.array(ed)/np.array(size)
            # scores_df["sagittal_profile"]=sagittal_profile
            # scores_df["frontal_profile"]=frontal_profile
            # scores_df["axial_profile"]=axial_profile
            scores_df["sagittal_reg"]=sagittal_reg
            scores_df["frontal_reg"]=frontal_reg
            scores_df["axial_reg"]=axial_reg
            scores_df["sagittal_da_profile"]=sagittal_da_profile
            scores_df["frontal_da_profile"]=frontal_da_profile
            scores_df["axial_da_profile"]=axial_da_profile      
                      
            
            
            scores_df.to_csv('eval_score_val'+str(val)+'_'+date.today().isoformat()+'T'+str(datetime.today().hour)+ model+csv_name+'.csv')
            fig, ax = plt.subplots(figsize=(10,6))
            
            ax.plot(mean_dice_model, label="Mean Dice")
            ax.set_ylim(0.6,0.95)
            ax.set_xlabel("Number of models used")
            ax.set_ylabel("Dice score")
            ax.plot(tumor_core, label='Tumor Core')
            ax.plot(whole_tumor, label='Whole tumor')
            ax.plot(enhancing_tumor,label='Enhancing tumor')
            ax.legend()
            ax.set_title("Dice score change with number of models")
            fig.tight_layout()
            
            plt.savefig("Dice"+ model_names[0]+str(plot))
            
            # print('mean_dice', mean_dice) 
        else:          
            models=[]
            for i,name in enumerate(model_names):
                if model=="UNet":
                     model=UNet(
                        spatial_dims=3,
                        in_channels=4,
                        out_channels=3,
                        channels=(64,128,256,512,1024),
                        strides=(2,2,2,2)
                        ).to(device)
                elif model=="SegResNet":
                    model = SegResNet(
                        blocks_down=[1, 2, 2, 4],
                        blocks_up=[1, 1, 1],
                        init_filters=32,
                        norm="instance",
                        in_channels=4,
                        out_channels=3,
                        upsample_mode=UpsampleMode[upsample]    
                        ).to(device)

                else:
                    model = locals() [model](4,3).to(device)
                
                model=torch.nn.DataParallel(model)
                
                model.load_state_dict(torch.load(eval_path+'/'+name),strict=False)
                model.eval()
                models.append(model)
                
            num_models=len(models)  
            if avgmodel:
                for key in model.state_dict().keys():
                    for i in models[:-1]:
                        model.state_dict()[key]=(i.state_dict()[key]+model.state_dict()[key] )
                    model.state_dict()[key]=model.state_dict()[key]/num_models
                torch.save(
                    model.state_dict(),
                    os.path.join("./saved models", date.today().isoformat()+'T'+str(datetime.today().hour)+ model+"zoo_avg"))
                    
                print("saved zoo model")
                
                ensemble = 0
                load_name=date.today().isoformat()+'T'+str(datetime.today().hour)+ model+"zoo_avg"    
                                

            mean_post_transforms = Compose(
                [
                    EnsureTyped(keys=["pred"+str(i) for i in range(len(models))]), #gives pred0..pred1...
                    ## SplitChanneld(keys=["pred"+str(i) for i in range(10)]),
                    Activationsd(keys=["pred"+str(i) for i in range(len(models))], sigmoid=True),
                    MeanEnsembled(
                        keys=["pred"+str(i) for i in range(len(models))], 
                        output_key="pred",
                      ##  # in this particular example, we use validation metrics as weights
                      ### weights=wts,
                    ),
                    Activationsd(keys="pred", sigmoid=True),
                    AsDiscreted(keys="pred", threshold=0.2),
                ]
            ) 
                               
            md,tc,wt,et,haus1,pred_size1,_,_,_,_,_,_,_,_,_,_,_=ensemble_evaluate(mean_post_transforms, models)
            mean_dice.append(md)
            tumor_core.append(tc)
            whole_tumor.append(wt)
            enhancing_tumor.append(et)
    elif ensemble==0:
        
        # model.load_state_dict(torch.load(eval_path),strict=False)
        model.load_state_dict(torch.load(eval_path, map_location=lambda storage, loc: storage.cuda(0)), strict=True)
        # model.to(device)
        model.eval()

        with torch.no_grad():

            for test_data in test_loader: # each image
                test_inputs = test_data["image"].to(device) # pass to gpu
                test_labels=test_data["label"].to(device)
                
                test_outputs=inference(test_inputs,model)
                test_outputs=[post_trans(i) for i in decollate_batch(test_outputs)] 
                # test_data["pred"] = sliding_window_inference(
                    # inputs=test_inputs,
                    # roi_size=(192,192, 144),
                    # sw_batch_size=batch_size,
                    # predictor=model,
                    # overlap=0,
                # )#inference(test_inputs) #perform inference
                #print(test_data["pred"].shape)
                # test_data=[post_transforms(i) for i in decollate_batch(test_data)] #reverse the transform and get sigmoid then binarise
                # test_outputs, test_labels =  from_engine(["pred", "label"])(test_data) # create list of images and labels
              
                
                #print("test outputs",test_outputs[0].shape)
                # test_outputs=[i.to(device) for i in test_outputs]
                # test_labels=[i.to(device) for i in test_labels]
                # dice_score=(2*torch.sum( test_outputs[0].flatten()*test_labels[0].flatten()))/( test_outputs[0].sum()+test_labels[0].sum())
                # print("dice",dice_score)
                
                dice_metric(y_pred=test_outputs, y=test_labels)
                dice_metric_batch(y_pred=test_outputs, y=test_labels)            
            metric_org = dice_metric.aggregate().item()
            
            metric_batch_org = dice_metric_batch.aggregate()

            dice_metric.reset()
            dice_metric_batch.reset()

        metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()

        print("Metric on original image spacing: ", metric_org)
        print(f"metric_tc: {metric_tc:.4f}", f"   metric_wt: {metric_wt:.4f}", f"   metric_et: {metric_et:.4f}")
     




