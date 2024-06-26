from typing import List, Optional, Sequence, Tuple, Union
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform
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
from monai.utils import TransformBackends, convert_data_type, deprecated_arg, ensure_tuple, look_up_option
import pandas as pd
# print(pandas.__version__)
import nibabel as nb
import re
import numpy as np
from monai.handlers import MeanDice, StatsHandler, ValidationHandler, from_engine,HausdorffDistance
from monai.inferers import SimpleInferer, SlidingWindowInferer
import numpy as np
import skimage
from Training.network import model

import scipy.ndimage as ndi

from monai.losses import DiceLoss

import torch
import gc

import os
from monai.transforms import (
MapTransform
)
from eval_config import *
from Evaluation.evaluation import inference
from Input.dataset import BratsDataset
from Input.config import root_dir,batch_size,workers
from Input.localtransforms import test_transforms0,post_trans,train_transform,val_transform


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npy', '.gz'
]
device = torch.device("cuda:0")
# A source: Nvidia HDGAN
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# makes a list of all image paths inside a directory
def make_dataset(data_dir):
    all_files = []
    images=[]
    labels=[]
    im_temp=[]
    assert os.path.isdir(data_dir), '%s is not a valid directory' % data_dir
    
    for root, fol, _ in sorted(os.walk(data_dir)): # list folders and root
        for folder in fol:                    # for each folder
             path=os.path.join(root, folder)  # combine root path with folder path
             for root1, _, fnames in os.walk(path):       #list all file names in the folder         
                for f in fnames:                          # go through each file name
                    fpath=os.path.join(root1,f)
                    if is_image_file(f):                  # check if expected extension
                        if re.search("seg",f):            # look for the label files- have'seg' in the name 
                            labels.append(fpath)
                        else:
                            im_temp.append(fpath)         # all without seg are image files, store them in a list for each folder
                if im_temp:
                    images.append(im_temp)                    # add image files for each folder to a list
                    im_temp=[]
    return images, labels
    
    # A source: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 1 and label 4 to construct TC
            result.append(np.logical_or(d[key] == 1, d[key] == 4))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 4), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 4)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d
        
class Ensemble:
    @staticmethod
    def get_stacked_torch(img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> torch.Tensor:
        """Get either a sequence or single instance of np.ndarray/torch.Tensor. Return single torch.Tensor."""
        if isinstance(img, Sequence) and isinstance(img[0], np.ndarray):
            print("converting pred list to tensor")
            img = [torch.as_tensor(i) for i in img]
        elif isinstance(img, np.ndarray):
            print("converting pred list to tensor")  
            img = torch.as_tensor(img)
        # else:
            # print("conversion wasn't necessary")
        
        out: torch.Tensor = torch.stack(img) if isinstance(img, Sequence) else torch.unsqueeze(img ,dim=0) # type: ignore
        return out

    @staticmethod
    def post_convert(img: torch.Tensor, orig_img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> NdarrayOrTensor:
        orig_img_ = orig_img[0] if isinstance(orig_img, Sequence) else orig_img
        out, *_ = convert_to_dst_type(img, orig_img_)
        return out
        

class ConfEnsemble(Ensemble, Transform):


    backend = [TransformBackends.TORCH]

    def __init__(self, weights: Optional[Union[Sequence[float], NdarrayOrTensor]] = None) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.float) if weights is not None else None

    def __call__(self, img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> NdarrayOrTensor:
        img_ = self.get_stacked_torch(img)
        if self.weights is not None:
            self.weights = self.weights.to(img_.device)
            shape = tuple(self.weights.shape)
            for _ in range(img_.ndimension() - self.weights.ndimension()):
                shape += (1,)
            weights = self.weights.reshape(*shape)

            img_ = img_ * weights / weights.mean(dim=0, keepdim=True)
            
        
        img_=torch.sort(img_,dim=0,descending=True).values #img is a tensor        
        
        # print(img_[0,2,100,100,70],img_[1,2,100,100,70],img_[2,2,100,100,70],img_[3,2,100,100,70],img_[4,2,100,100,70])
        out_pt = torch.mean(img_[0:10,:,:,:,:], dim=0)#[0:3,:,:,:,:]
        # print(out_pt.shape)
        x=self.post_convert(out_pt, img)
        
       
        return x
        
class Ensembled(MapTransform):
   

    backend = list(set(ConfEnsemble.backend) )

    def __init__(
        self, keys,ensemble,
        output_key = None,
        allow_missing_keys = False,
    ) -> None:
   
        super().__init__(keys, allow_missing_keys)
        if not callable(ensemble):
            raise TypeError(f"ensemble must be callable but is {type(ensemble).__name__}.")
        self.ensemble = ensemble
        if len(self.keys) > 1 and output_key is None:
            raise ValueError("Incompatible values: len(self.keys) > 1 and output_key=None.")
        self.output_key = output_key if output_key is not None else self.keys[0]

    def __call__(self, data):
        d = dict(data)
        
        if len(self.keys) == 1 and self.keys[0] in d:
            items = d[self.keys[0]]
        else:
            items = [d[key] for key in self.key_iterator(d)]

        if len(items) > 0:
            d[self.output_key] = self.ensemble(items)

        return d
        
class ConfEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MeanEnsemble`.
    """

    backend = ConfEnsemble.backend

    def __init__(
        self,
        keys,
        output_key = None,
        weights = None,
    ) -> None:

        ensemble = ConfEnsemble(weights=weights)
        super().__init__(keys, ensemble, output_key)

def ensemble_evaluate(post_transforms, models,test_loader):
    
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
                roi_size=(240,240,152), sw_batch_size=4, overlap=0
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
    
    

class TestDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        data=make_dataset(data_dir)
        self.image_list=data[0]  
        self.label_list=data[1] 
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
        # print(item_dict)
        
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
            
# Function to calculate GLCM properties
def calculate_glcm_features(image, distances, angles, levels):
    glcm = skimage.feature.greycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    feature_dict = {}
    for prop in properties:
        value = skimage.feature.greycoprops(glcm, prop)[0, 0]
        feature_dict[prop] = value
    return feature_dict
    
 # Convert list to a numpy array and return


def mask_feat(imagef,gt_used,levels=256):

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
    results=[]
    for image_paths,gt in zip(imagef,gt_used):
    
        mask=nb.load(gt).get_fdata()
    
        lot_size=len(np.nonzero(mask)[0])
        
        et_mask=np.where(mask==3,1,0)
        
        core_mask=np.where(mask==1,1,0)
        edema_mask=np.where(mask==2,1,0)
        
        tc_mask=et_mask+core_mask
        wt_mask=et_mask+edema_mask+core_mask
        
        
        etumour=np.where(mask==3,1,0).sum()
        
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
        
       
        # Calculate the centroid of the mask -different to largest wt area
        centroid = np.round(ndi.center_of_mass(mask_bi)).astype(int)
        
        subject_features = {'mask_path': gt}
        subject_features['a_centroid']= centroid[2]
        subject_features['c_centroid']= centroid[1]
        subject_features['s_centroid']= centroid[0]
        
        
        areas_a = mask_bi.sum(axis=(0, 1))  # Sum along the x and y axes
        a_argmax = areas_a.argmax()
        subject_features['a_argmax'] =  a_argmax
        areas_s = mask_bi.sum(axis=(1, 2))
        s_argmax = areas_s.argmax()
        subject_features['s_argmax']= s_argmax 
        areas_c = mask_bi.sum(axis=(0, 2))
        c_argmax = areas_c.argmax()
        subject_features['c_argmax']= c_argmax 
        
        
        for i,image_path in enumerate(image_paths):
            image_data=nb.load(image_paths[i]).get_fdata()
            image_slice_axial = image_data[:, :, a_argmax]
            image_slice_coronal = image_data[:, c_argmax, :]
            image_slice_sagittal = image_data[s_argmax, :, :]
            
            image_slice_axial = ((image_slice_axial - image_slice_axial.min()) * (levels - 1) / (image_slice_axial.max() - image_slice_axial.min())).astype(np.uint8)
            
            image_slice_coronal = ((image_slice_coronal - image_slice_coronal.min()) * (levels - 1) / (image_slice_coronal.max() - image_slice_coronal.min())).astype(np.uint8)
            
            image_slice_sagittal = ((image_slice_sagittal - image_slice_sagittal.min()) * (levels - 1) / (image_slice_sagittal.max() - image_slice_sagittal.min())).astype(np.uint8)
            
            features = calculate_glcm_features(image_slice_axial, distances=[5], angles=[0], levels=256)
            
            for key, value in features.items():
                subject_features[f'axial{i}_{key}'] = value
                
            features = calculate_glcm_features(image_slice_coronal, distances=[5], angles=[0], levels=256)
            
            for key, value in features.items():
                subject_features[f'coronal{i}_{key}'] = value
                
            features = calculate_glcm_features(image_slice_sagittal, distances=[5], angles=[0], levels=256)
            
            for key, value in features.items():
                subject_features[f'sagittal{i}_{key}'] = value
            
        results.append(subject_features)         
                        
        
        
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
    
    df = pd.DataFrame(results)
    print('df.shape',df.shape)
    df = df.assign(**{        
        'tumour_core': tumour_core,
        'enhancing tumor': ent,
        'edema': ed,
        'size': size,
        'sagittal_profile_tc': sagittal_profile_tc,
        'frontal_profile_tc': frontal_profile_tc,
        'axial_profile_tc': axial_profile_tc,
        'sagittal_profile_wt': sagittal_profile_wt,
        'frontal_profile_wt': frontal_profile_wt,
        'axial_profile_wt': axial_profile_wt,
        'sagittal_profile_et': sagittal_profile_et,
        'frontal_profile_et': frontal_profile_et,
        'axial_profile_et': axial_profile_et,
        'sagittal_reg': sagittal_reg,
        'frontal_reg': frontal_reg,
        'axial_reg': axial_reg,
        'sagittal_da_profile': sagittal_da_profile,
        'frontal_da_profile': frontal_da_profile,
        'axial_da_profile': axial_da_profile,
        'reg_score_avg': reg_score_avg,
        'da_prof_avg': da_prof_avg,
        })
    print('df.shape',df.shape)
    
    return df
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        