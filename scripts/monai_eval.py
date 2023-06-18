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

from config import *
from brats_transforms import *
import config



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


    ##################~~~~~~~~~~~~~~~~~~~~~~~~~Model Definition~~~~~~~~~~~~~~~~~~~~~~~#################
 

    # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
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
        

    # define inference method
    def inference(input):

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(192,192, 144),
                sw_batch_size=1,
                predictor=model,
                overlap=0,
            )

        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)


    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True





    indexes=np.arange(max_samples)
    fold=int(max_samples/10)
    
    class TestDataset(Dataset):
        def __init__(self,data_dir,transform=None):
            self.image_list=make_dataset(data_dir)[0]  
            self.label_list=make_dataset(data_dir)[1] 
            self.transform=transform
            
        def __len__(self):
    #         return len(os.listdir(self.label_dir))
            if val==1:
                return len(self.label_list)
            else:
                return min(max_samples,len(self.label_list))
        
        def __getitem__(self,idx):
            image=self.image_list[idx]
        
            label=self.label_list[idx] 

                
            item_dict={"image":image,"label":label}
            
            test_list=dict()
            if self.transform:
                item_dict={"image":image,"label": label}
                
                test_list=self.transform(item_dict)
                
            
            return test_list
            
    # val_indices=np.random.choice(np.arange(100),30,replace=False)
    
    def countsize(y_pred,y):
        
        n_len = len(y_pred.shape)
        reduce_axis = list(range(2, n_len)) # each channel gives out one value
        
        y_pred = y_pred.float()
        
        all_things=[]
        predsize=torch.count_nonzero(y_pred,dim=reduce_axis).float()     #shape=[1,3,...]
        
        diceP_indSagittal_tc=[]
        diceP_indFrontal_tc=[]
        diceP_indAxial_tc=[]
        
        diceP_indSagittal_wt=[]
        diceP_indFrontal_wt=[]
        diceP_indAxial_wt=[]
        
        diceP_indSagittal_et=[]
        diceP_indFrontal_et=[]
        diceP_indAxial_et=[]
        
        for x in range(1,y_pred.shape[2]):
            dice_sag_tc=(2*(y_pred[0,0,x-1,:,:]*y_pred[0,0,x,:,:]).sum())\
                        /(y_pred[0,0,x-1,:,:].sum()+y_pred[0,0,x,:,:].sum()+0.001)                    
            dice_sag_wt=(2*(y_pred[0,1,x-1,:,:]*y_pred[0,1,x,:,:]).sum())\
                        /(y_pred[0,1,x-1,:,:].sum()+y_pred[0,1,x,:,:].sum()+0.001)                       
            dice_sag_et=(2*(y_pred[0,2,x-1,:,:]*y_pred[0,2,x,:,:]).sum())\
                        /(y_pred[0,2,x-1,:,:].sum()+y_pred[0,2,x,:,:].sum()+0.001)
            if dice_sag_wt>0:
                diceP_indSagittal_wt.append(dice_sag_wt)
            if dice_sag_et>0:
                diceP_indSagittal_et.append(dice_sag_et)
            if dice_sag_tc>0:
                diceP_indSagittal_tc.append(dice_sag_tc)
            
        for yi in range(1,y_pred.shape[3]):
            dice_fr_tc=(2*(y_pred[0,0,:,yi-1,:]*y_pred[0,0,:,yi,:]).sum())\
                        /(y_pred[0,0,:,yi-1,:].sum()+y_pred[0,0,:,yi,:].sum()+0.001)
                        
            dice_fr_wt=(2*(y_pred[0,1,:,yi-1,:]*y_pred[0,1,:,yi,:]).sum())\
                        /(y_pred[0,1,:,yi-1,:].sum()+y_pred[0,1,:,yi,:].sum()+0.001)
                        
            dice_fr_et=(2*(y_pred[0,2,:,yi-1,:]*y_pred[0,2,:,yi,:]).sum())\
                        /(y_pred[0,2,:,yi-1,:].sum()+y_pred[0,2,:,yi,:].sum()+0.001)
            if dice_fr_tc>0:            
                diceP_indFrontal_tc.append(dice_fr_tc)
            if dice_fr_wt>0:
                diceP_indFrontal_wt.append(dice_fr_wt)
            if dice_fr_et>0:
                diceP_indFrontal_et.append(dice_fr_et)
            
        for z in range(1,y_pred.shape[4]):
            dice_ax_tc=(2*(y_pred[0,0,:,:,z-1]*y_pred[0,0,:,:,z]).sum())\
                        /(y_pred[0,0,:,:,z-1].sum()+y_pred[0,0,:,:,z].sum()+0.001)
                        
            dice_ax_wt=(2*(y_pred[0,1,:,:,z-1]*y_pred[0,1,:,:,z]).sum())\
                        /(y_pred[0,1,:,:,z-1].sum()+y_pred[0,1,:,:,z-1].sum()+0.001)
                        
            dice_ax_et=(2*(y_pred[0,2,:,:,z-1]*y_pred[0,2,:,:,z]).sum())\
                        /(y_pred[0,2,:,:,z-1].sum()+y_pred[0,2,:,:,z].sum()+0.001)
                        
            if dice_ax_tc>0:
                diceP_indAxial_tc.append(dice_ax_tc)
            if dice_ax_wt>0:
                diceP_indAxial_wt.append(dice_ax_wt)
            if dice_ax_et>0:
                diceP_indAxial_et.append(dice_ax_et)
        
        diceP_indSagittal_tc_=torch.tensor(diceP_indSagittal_tc).mean()
        diceP_indSagittal_wt_=torch.tensor(diceP_indSagittal_wt).mean()
        diceP_indSagittal_et_=torch.tensor(diceP_indSagittal_et).mean()        
        
        diceP_indFrontal_tc_=torch.tensor(diceP_indFrontal_tc).mean()
        diceP_indFrontal_wt_=torch.tensor(diceP_indFrontal_wt).mean()
        diceP_indFrontal_et_=torch.tensor(diceP_indFrontal_et).mean()
        
        diceP_indAxial_tc_=torch.tensor(diceP_indAxial_tc).mean()
        diceP_indAxial_wt_=torch.tensor(diceP_indAxial_wt).mean()
        diceP_indAxial_et_=torch.tensor(diceP_indAxial_et).mean()
        
        sag=torch.tensor([diceP_indSagittal_tc_,diceP_indSagittal_wt_,diceP_indSagittal_et_]).float().view(1,3).to(device)
        
        front=torch.tensor([diceP_indFrontal_tc_,diceP_indFrontal_wt_,diceP_indFrontal_et_]).float().view(1,3).to(device)
        axial=torch.tensor([diceP_indAxial_tc_,diceP_indAxial_wt_,diceP_indAxial_et_]).float().view(1,3).to(device)
        
        predsize=torch.cat((predsize,sag),0)
        predsize=torch.cat((predsize,front),0)       
        predsize=torch.cat((predsize,axial),0).view(1,4,3)
        
        return predsize
        
    class CountSize(CumulativeIterationMetric):
        def __init__(self,
        reduction=MetricReduction.MEAN_CHANNEL,
        output_transform=lambda x:x,
        get_not_nans = False,
        ignore_empty = True,
        ):        
            super().__init__()
            self.reduction=reduction
            self.get_not_nans = get_not_nans
            self.ignore_empty = ignore_empty
        def _compute_tensor(self,y_pred,y):
            return countsize(y_pred,y)
            
        def aggregate(self):
            data=self.get_buffer()
            
            if not isinstance(data, torch.Tensor):
                raise ValueError("the data to aggregate must be PyTorch Tensor.")
            
            f, not_nans = do_metric_reduction(data, self.reduction)
            return (f, not_nans) if self.get_not_nans else f
    class PredSize(IgniteMetric):
        def __init__(self,
            reduction=MetricReduction.MEAN_CHANNEL,
            output_transform=lambda x:x,
            save_details=True
            ):        
            metric_fn=CountSize(
            reduction=reduction)
            super().__init__(metric_fn=metric_fn,output_transform=output_transform,save_details=save_details)
            
   
        
    if val==1:
        val_indices=np.random.choice(np.arange(1000),max_samples,replace=False)
        test_ds=TestDataset("./RSNA_ASNR_MICCAI_BraTS2021_TrainingData",transform=test_transforms0)
        test_ds=Subset(test_ds,val_indices)
        gt_all=make_dataset("./RSNA_ASNR_MICCAI_BraTS2021_TrainingData")[1]
        gt_used=[gt_all[i] for i in val_indices]
        
        #print("list of input gt files",make_dataset("./RSNA_ASNR_MICCAI_BraTS2021_TestData")[1])
    else:
        val_indices=np.random.choice(np.arange(200),max_samples,replace=False)
        test_ds=TestDataset("./RSNA_ASNR_MICCAI_BraTS2021_TestData",transform=test_transforms0)
        test_ds=Subset(test_ds,val_indices)
        gt_all=make_dataset("./RSNA_ASNR_MICCAI_BraTS2021_TestData")[1]
        gt_used=[gt_all[i] for i in val_indices]
    
    size=[]
    ent=[]
    tumour_core=[]
    ed=[]
    
    
    
    
    all_sizes=dict()
    all_areasSagittal=dict()
    all_areasFrontal=dict()
    all_areasAxial=dict()
    size_factors=dict()
    
    dice_profSagittal_et=[]
    dice_profFrontal_et=[]
    dice_profAxial_et=[]
    
    dice_profSagittal_tc=[]
    dice_profFrontal_tc=[]
    dice_profAxial_tc=[]
    
    dice_profSagittal_wt=[]
    dice_profFrontal_wt=[]
    dice_profAxial_wt=[]
    
    reg_score_sagittal=[]
    reg_score_frontal=[]
    reg_score_axial=[]
    da_profSagittal=[]
    da_profFrontal=[]
    da_profAxial=[]
    prev_area=0
    kernel=np.ones((3,3),np.uint8)
    for gt in gt_used:
        mask=nb.load(gt).get_fdata()
        lot_size=len(np.nonzero(mask)[0])
        
        et_mask=np.where(mask==4,1,0)
        
        core_mask=np.where(mask==1,1,0)
        edema_mask=np.where(mask==2,1,0)
        
        tc_mask=et_mask+core_mask
        wt_mask=et_mask+edema_mask+core_mask
        
        
        etumour=np.where(mask==4,1,0).sum()
        
        core=np.where(mask==1,1,0).sum()
        edema=np.where(mask==2,1,0).sum()
        
        tumour_core_ins=etumour+core
        
        
        ent.append(etumour)
        tumour_core.append(tumour_core_ins)
        ed.append(edema)         
        
        
        size.append(lot_size)
        
        
        
        area_storeSagittal=[]
        area_storeFrontal=[]
        area_storeAxial=[]
        areas=[]        
        diceP_indSagittal_tc=[]
        diceP_indFrontal_tc=[]
        diceP_indAxial_tc=[]
        
        diceP_indSagittal_wt=[]
        diceP_indFrontal_wt=[]
        diceP_indAxial_wt=[]
        
        diceP_indSagittal_et=[]
        diceP_indFrontal_et=[]
        diceP_indAxial_et=[]
        
        reg_indSagittal=[]
        reg_indFrontal=[]
        reg_indAxial=[]
        daP_indSagittal=[]
        daP_indFrontal=[]
        daP_indAxial=[]
        
        
        
        mask_bi=np.where(mask,1,mask)
        for x in range(mask.shape[0]):
            area_slice=len(np.nonzero(mask[x,:,:])[0])
            area_storeSagittal.append(area_slice)
            perimeter=len(np.nonzero(cv2.erode(mask[x,:,:],kernel))[0])
            rep_rad=np.sqrt(area_slice/np.pi)
            reg_score=perimeter/(2*np.pi*rep_rad + 0.001)
            reg_indSagittal.append(reg_score)
            
            if x >0:
                dice_sag_tc=(2*(tc_mask[x-1,:,:]*tc_mask[x,:,:]).sum())\
                            /(tc_mask[x-1,:,:].sum()+tc_mask[x,:,:].sum()+0.001)
                diceP_indSagittal_tc.append(dice_sag_tc)
                
                dice_sag_wt=(2*(wt_mask[x-1,:,:]*wt_mask[x,:,:]).sum())\
                            /(wt_mask[x-1,:,:].sum()+wt_mask[x,:,:].sum()+0.001)
                diceP_indSagittal_wt.append(dice_sag_wt)
                
                dice_sag_et=(2*(et_mask[x-1,:,:]*et_mask[x,:,:]).sum())\
                            /(et_mask[x-1,:,:].sum()+et_mask[x,:,:].sum()+0.001)
                diceP_indSagittal_et.append(dice_sag_et)
                
                prev_area=len(np.nonzero(mask[x-1,:,:])[0])
                darea=2*abs(area_slice-prev_area)/(area_slice+prev_area+0.001)
                daP_indSagittal.append(darea)
                    
        for y in range(mask.shape[1]):
            area_slice=len(np.nonzero(mask[:,y,:])[0])
            area_storeFrontal.append(area_slice)
            perimeter=len(np.nonzero(cv2.erode(mask[:,y,:],kernel))[0])
            rep_rad=np.sqrt(area_slice/np.pi)
            reg_score=perimeter/(2*np.pi*rep_rad + 0.001)
            reg_indFrontal.append(reg_score)
            if y>0:
                dice_fr_tc=(2*(tc_mask[:,y-1,:]*tc_mask[:,y,:]).sum())\
                    /(tc_mask[:,y-1,:].sum()+tc_mask[:,y,:].sum()+0.001)
                diceP_indFrontal_tc.append(dice_fr_tc)
                
                dice_fr_wt=(2*(wt_mask[:,y-1,:]*wt_mask[:,y,:]).sum())\
                    /(wt_mask[:,y-1,:].sum()+wt_mask[:,y,:].sum()+0.001)
                diceP_indFrontal_wt.append(dice_fr_wt)
                
                dice_fr_et=(2*(et_mask[:,y-1,:]*et_mask[:,y,:]).sum())\
                    /(et_mask[:,y-1,:].sum()+et_mask[:,y,:].sum()+0.001)
                diceP_indFrontal_et.append(dice_fr_et)
                
                
                prev_area=len(np.nonzero(mask[:,y-1,:])[0])
                darea=2*abs(area_slice-prev_area)/(area_slice+prev_area+0.001)
                daP_indFrontal.append(darea)
                    
        for z in range(mask.shape[2]):
            area_slice=len(np.nonzero(mask[:,:,z])[0])
            # print(area_slice,'area_slice')
            area_storeAxial.append(area_slice) 
            perimeter=len(np.nonzero(cv2.erode(mask[:,:,z],kernel))[0])
            rep_rad=np.sqrt(area_slice/np.pi)
            reg_score=perimeter/(2*np.pi*rep_rad + 0.001)
            reg_indAxial.append(reg_score)
            if z>0:
                dice_ax_tc=(2*(tc_mask[:,:,z-1]*tc_mask[:,:,z]).sum())\
                    /(tc_mask[:,:,z-1].sum()+tc_mask[:,:,z].sum()+0.001)
                diceP_indAxial_tc.append(dice_ax_tc)
                
                dice_ax_wt=(2*(wt_mask[:,:,z-1]*wt_mask[:,:,z]).sum())\
                    /(wt_mask[:,:,z-1].sum()+wt_mask[:,:,z].sum()+0.001)
                diceP_indAxial_wt.append(dice_ax_wt)
                
                dice_ax_et=(2*(et_mask[:,:,z-1]*et_mask[:,:,z]).sum())\
                    /(et_mask[:,:,z-1].sum()+et_mask[:,:,z].sum()+0.001)
                diceP_indAxial_et.append(dice_ax_et)
                
                prev_area=len(np.nonzero(mask[:,:,z-1])[0])
                darea=2*abs(area_slice-prev_area)/(area_slice+prev_area+0.001)
                daP_indAxial.append(darea)
                
        all_areasSagittal[gt[-5:]]=area_storeSagittal # areas in all axes
        all_areasFrontal[gt[-5:]]=area_storeFrontal
        all_areasAxial[gt[-5:]]=area_storeAxial
        all_sizes[gt[-5:]]=lot_size
        
        dice_profSagittal_tc.append(diceP_indSagittal_tc)
        dice_profFrontal_tc.append(diceP_indFrontal_tc)
        dice_profAxial_tc.append(diceP_indAxial_tc)
        
        dice_profSagittal_wt.append(diceP_indSagittal_wt)
        dice_profFrontal_wt.append(diceP_indFrontal_wt)
        dice_profAxial_wt.append(diceP_indAxial_wt)
        
        dice_profSagittal_et.append(diceP_indSagittal_et)
        dice_profFrontal_et.append(diceP_indFrontal_et)
        dice_profAxial_et.append(diceP_indAxial_et)
        
        reg_score_axial.append(reg_indAxial)
        reg_score_frontal.append(reg_indFrontal)
        reg_score_sagittal.append(reg_indSagittal)
        da_profSagittal.append(daP_indSagittal)
        da_profAxial.append(daP_indAxial)
        da_profFrontal.append(daP_indFrontal)
        
    #Consecutive dice score
    sagittal_profile_tc=np.ma.masked_equal(np.array(dice_profSagittal_tc),0).mean(axis=1)#to take only non zero values for average
    frontal_profile_tc=np.ma.masked_equal(np.array(dice_profFrontal_tc),0).mean(axis=1)
    axial_profile_tc=np.ma.masked_equal(np.array(dice_profAxial_tc),0).mean(axis=1)
    
    sagittal_profile_wt=np.ma.masked_equal(np.array(dice_profSagittal_wt),0).mean(axis=1)#to take only non zero values for average
    frontal_profile_wt=np.ma.masked_equal(np.array(dice_profFrontal_wt),0).mean(axis=1)
    axial_profile_wt=np.ma.masked_equal(np.array(dice_profAxial_wt),0).mean(axis=1)
    
    sagittal_profile_et=np.ma.masked_equal(np.array(dice_profSagittal_et),0).mean(axis=1)#to take only non zero values for average
    frontal_profile_et=np.ma.masked_equal(np.array(dice_profFrontal_et),0).mean(axis=1)
    axial_profile_et=np.ma.masked_equal(np.array(dice_profAxial_et),0).mean(axis=1)
    
   
    
    # avg_dice_profile=(sagittal_profile+frontal_profile+axial_profile)/3

    #consecutive reg score
    sagittal_reg=np.ma.masked_equal(np.array(reg_score_sagittal),0).mean(axis=1)
    frontal_reg=np.ma.masked_equal(np.array(reg_score_frontal),0).mean(axis=1)
    axial_reg=np.ma.masked_equal(np.array(reg_score_axial),0).mean(axis=1)
    
    reg_score_avg=(sagittal_reg+frontal_reg+axial_reg)/3

    #delta area between slices 
    sagittal_da_profile=np.ma.masked_equal(np.array(da_profSagittal),0).mean(axis=1)
    frontal_da_profile=np.ma.masked_equal(np.array(da_profFrontal),0).mean(axis=1)
    axial_da_profile=np.ma.masked_equal(np.array(da_profAxial),0).mean(axis=1)
    
    da_prof_avg=(sagittal_da_profile+frontal_da_profile+axial_da_profile)/3
        
        
        
    

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers) # this should return 10 different instances

    # print("input type",type(next(iter(test_loader))))





    post_transforms = Compose([
        EnsureTyped(keys=["pred","label"]), 
        Activationsd(keys="pred", sigmoid=True),
        Invertd(
            keys="pred",
            transform=test_transforms0,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            # meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True, #this sends to GPU so removing will cause problems
            device=device
        ), # inversal is only done on the prediction? yes with the specified key
        # ToTensorD(keys=["pred","label"]),
        
        AsDiscreted(keys="pred", threshold=0.5),
        SaveImaged(keys=["pred","label"],meta_keys="pred_meta_dict",output_dir="./ssensemblemodels0922/outputs/",resample=False)
    ])
    post_pred= Compose([EnsureType(),Activations(sigmoid=True), AsDiscrete(threshold=0.2)])
    post_label = Compose([EnsureType()])
    
    print(type(model))

    if ensemble==1:
    
        if config.model=="UNet":
            model_names= os.listdir(eval_path)
            #["UNetep100rs4C","UNetep99rs4C","UNetep98rs4C","UNetep97rs4C","UNetep96rs4C"]


            print(model_names)
            # wts=[0.69,0.69,0.78,0.72,0.62,0.7,0.7,0.7,0.75,0.67]
       
        elif config.model=="SegResNet":
            model_names=os.listdir(eval_path)#'/scratch/a.bip5/BraTS 2021/ssensemblemodels0922/Evaluation Folder1')#["SegResNetep100rs2C","SegResNetep99rs2C","SegResNetep98rs2C","SegResNetep97rs2C","SegResNetep96rs2C"]
            
           





            
            print(model_names)
            # wts=[0.7401,0.7506,0.6364,0.7484, 0.6725,0.7707,0.7219,0.7439,0.8003,0.7458]#[0.5651,0.5252,0.5537,0.5137,0.5744,0.4862,0.5255,0.5559,0.5755,0.5060]
        

        else:
            print("No SISA yet")

        def ensemble_evaluate(post_transforms, models):
            
            for i in range(len(models)):
                key_val_metric={
                    f"test_mean_dice": MeanDice(
                        include_background=True,
                        output_transform=from_engine([f"pred{i}", "label"]),
                        reduction='mean'
                    )
                }
                additional_metrics={ 
                    f"Channelwise": MeanDice(
                        include_background=True,
                        output_transform=from_engine([f"pred{i}", "label"]),
                        reduction="mean_batch"
                    ),
                    f"Hausdorff": HausdorffDistance(
                        include_background=False,                      
                        output_transform=from_engine([f"pred{i}", "label"]),
                        reduction='mean_channel',
                        percentile=95
                    ),
                    f"pred size": PredSize(
                        output_transform=from_engine([f"pred{i}", 'label']),
                        reduction='none'
                    )
                }

                evaluator = EnsembleEvaluator(
                    device=device,
                    val_data_loader=test_loader,
                    pred_keys=[f"pred{i}"],
                    networks=[models[i]],
                    inferer=SlidingWindowInferer(
                        roi_size=(192,192, 144), sw_batch_size=4, overlap=0
                    ),
                    postprocessing=post_transforms,
                    key_val_metric=key_val_metric,
                    additional_metrics=additional_metrics
                )
                evaluator.run()

            
            # print("validation stats: ",evaluator.get_validation_stats())
            mean_dice=evaluator.state.metrics['test_mean_dice']#[:,1]#wt score
            pred_size=evaluator.state.metrics['pred size'][:,0,1] #whole tumor 
            pred_tc=evaluator.state.metrics['pred size'][:,0,0]
            pred_et=evaluator.state.metrics['pred size'][:,0,2]
            pred_dp_sag_tc=evaluator.state.metrics['pred size'][:,1,0]
            pred_dp_sag_wt=evaluator.state.metrics['pred size'][:,1,1]
            pred_dp_sag_et=evaluator.state.metrics['pred size'][:,1,2]
            pred_dp_fr_tc=evaluator.state.metrics['pred size'][:,2,0]
            pred_dp_fr_wt=evaluator.state.metrics['pred size'][:,2,1]
            pred_dp_fr_et=evaluator.state.metrics['pred size'][:,2,2]
            pred_dp_ax_tc=evaluator.state.metrics['pred size'][:,3,0]
            pred_dp_ax_wt=evaluator.state.metrics['pred size'][:,3,1]
            pred_dp_ax_et=evaluator.state.metrics['pred size'][:,3,2]
            hausdorff=evaluator.state.metrics['Hausdorff']
            tumor_core=evaluator.state.metrics["Channelwise"][0]
            whole_tumor=evaluator.state.metrics["Channelwise"][1]
            enhancing_tumor=evaluator.state.metrics["Channelwise"][2]
            print("Mean Dice:",evaluator.state.metrics['test_mean_dice'],"metric_tc:",float(evaluator.state.metrics["Channelwise"][0]),"whole tumor:",float(evaluator.state.metrics["Channelwise"][1]),"enhancing tumor:",float(evaluator.state.metrics["Channelwise"][2]))#jbc
            
            return mean_dice,tumor_core,whole_tumor,enhancing_tumor,hausdorff,pred_size,pred_tc,pred_et,pred_dp_sag_tc,pred_dp_sag_wt,pred_dp_sag_et,pred_dp_fr_tc,pred_dp_fr_wt,pred_dp_fr_et,pred_dp_ax_tc,pred_dp_ax_wt,pred_dp_ax_et
        
        
        

        
        
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
                   
                    if config.model=="UNet":
                         model=UNet(
                            spatial_dims=3,
                            in_channels=4,
                            out_channels=3,
                            channels=(64,128,256,512,1024),
                            strides=(2,2,2,2)
                            ).to(device)
                    elif config.model=="SegResNet":
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
               
                    if config.model=="UNet":
                         model1=UNet(
                            spatial_dims=3,
                            in_channels=4,
                            out_channels=3,
                            channels=(64,128,256,512,1024),
                            strides=(2,2,2,2)
                            ).to(device)
                    elif config.model=="SegResNet":
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
                        Activationsd(keys="pred", sigmoid=True),
                        AsDiscreted(keys="pred", threshold=0.2),
                    ]
                )
                                   
                
                
                md1,tc1,wt1,et1,haus1,pred_size1,_,_,_,_,_,_,_,_,_,_,_=ensemble_evaluate(mean_post_transforms1, models1)
                scores[model_steps1[0]]=md1
                scores['haus_'+ model_steps1[0]]=haus1
                if i==4:
                    md,tc,wt,et,haus,pred_size,pred_tc,pred_et,pred_dp_sag_tc,pred_dp_sag_wt,pred_dp_sag_et,pred_dp_fr_tc,pred_dp_fr_wt,pred_dp_fr_et,pred_dp_ax_tc,pred_dp_ax_wt,pred_dp_ax_et=ensemble_evaluate(mean_post_transforms, models)
                    #if val==1: # in case we want individualised score only for train data
                    scores["Ensemble"]=md
                    scores["HausEnsemble"]=haus
                if plot=1:
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
        
        model.load_state_dict(torch.load("./ssensemblemodels0922/Evaluation Folder 1/"+load_name),strict=False)
        # model.to(device)
        model.eval()

        with torch.no_grad():

            for test_data in test_loader: # each image
                test_inputs = test_data["image"].to(device) # pass to gpu
                test_labels=test_data["label"].to(device)
                test_data["pred"] = sliding_window_inference(
                    inputs=test_inputs,
                    roi_size=(192,192, 144),
                    sw_batch_size=batch_size,
                    predictor=model,
                    overlap=0,
                )#inference(test_inputs) #perform inference
                #print(test_data["pred"].shape)
                test_data=[post_transforms(i) for i in decollate_batch(test_data)] #reverse the transform and get sigmoid then binarise
                test_outputs, test_labels =  from_engine(["pred", "label"])(test_data) # create list of images and labels
              
                
                #print("test outputs",test_outputs[0].shape)
                test_outputs=[i.to(device) for i in test_outputs]
                test_labels=[i.to(device) for i in test_labels]
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
     




