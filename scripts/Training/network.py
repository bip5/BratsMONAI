import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')

# def uncache(exclude):
    # """Remove package modules from cache except excluded ones.
    # On next import they will be reloaded.

    # Args:
        # exclude (iter<str>): Sequence of module paths.
    # """
    # pkgs = []
    # for mod in exclude:
        # pkg = mod.split('.', 1)[0]
        # pkgs.append(pkg)

    # to_uncache = []
    # for mod in sys.modules:
        # if mod in exclude:
            # continue

        # if mod in pkgs:
           # to_uncache.append(mod)
           # continue

        # for pkg in pkgs:
            # if mod.startswith(pkg + '.'):
                # to_uncache.append(mod)
                # break

    # for mod in to_uncache:
        # del sys.modules[mod]
from Input.config import (
model_name,
load_save,
seed,
load_path,
upsample,
dropout,
init_filter_number,
num_layers,
num_filters
)
from Training.CustomActivation import LinSig,AllActivation
import torch
from torch import nn
from monai.networks.nets import UNet, SwinUNETR, SegResNetVAE
from Training.segresnetprj import SegResNet
from Training.custom_networks import SegResNetAtt,manUNet,WNet
from Training.deepFocus import DeepFocusCNN
from Training.hiddenfocus import HiddenFocus as DualFocus
from Training.scale_focus import ScaleFocus
from Training.layer_net import LayerNet
from monai.utils import UpsampleMode
import numpy as np
from monai.utils import set_determinism

import torch.nn.functional as F

device = torch.device("cuda:0")

np.random.seed(seed)
torch.cuda.manual_seed(seed)
set_determinism(seed=seed)    
torch.manual_seed(seed)

def create_model(model_name=model_name):
    if model_name=="manUNet":
        model=manUNet(4,3,i=init_filter_number)

    elif model_name=='WNet':
        model=WNet(4,3,i=init_filter_number)  
    elif model_name=="DeepFocus":
        model=DeepFocusCNN()
    elif model_name=="ScaleFocus":
        model=ScaleFocus(num_layers=num_layers,num_filters=num_filters)
    elif model_name=="DualFocus":
        model=DualFocus()
      
    elif model_name=="UNet":    
        model=UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(16,32,64,128,256),
            strides=(2,2,2,2)
            ).to(device)
    elif model_name=="LayerNet":
        model = LayerNet(        
            in_channels=4,
            out_channels=3,
            layers=5,
            units_per_layer=10
            ).to(device) 
    elif model_name=="SegResNetAtt":
        model = SegResNetAtt(        
            n_channels=4,
            n_classes=3,
            # init_filters=init_filter_number,
            ).to(device)        
    elif model_name=="SegResNet":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=init_filter_number,
            norm="instance",
            
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[upsample],
            dropout_prob=dropout
            ).to(device)
    elif model_name=="SegResNet_half":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            norm="instance",
            
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[upsample],
            dropout_prob=dropout
            ).to(device)
    elif model_name=="transformer":
        model=SwinUNETR(
        img_size=[192,192,128],
        in_channels=4,
        out_channels=3,
        ).to(device)
    elif model_name=="SegResNet_CA":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=32,
            norm="instance",
            act='LinSig',
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[upsample],
            dropout_prob=dropout
            ).to(device)
    elif model_name=="SegResNet_CA_half":    
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            norm="instance",
            act='LinSig',        
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[upsample],
            dropout_prob=dropout
            ).to(device)
    elif model_name=="SegResNet_CA_quarter":    
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=8,
            norm="instance",
            act='LinSig',        
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[upsample],
            dropout_prob=dropout
            ).to(device)  
    elif model_name=="SegResNet_CA2":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=32,
            norm="instance",
            act='AllActivation',
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[upsample],
            dropout_prob=dropout
            ).to(device)
    elif model_name=="SegResNet_CA2_half":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=32,
            norm="instance",
            act='AllActivation',
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[upsample],
            dropout_prob=dropout
            ).to(device)        
    elif model_name=="SegResNet_Flipper":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=32,
            norm="instance",
            
            in_channels=8,
            out_channels=3,
            upsample_mode=UpsampleMode[upsample],
            dropout_prob=dropout
            ).to(device)
    elif model_name=="SegResNetVAE":
        model = SegResNetVAE(
            input_image_size=(192,192,144),
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=init_filter_number,
            norm="instance",
            
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[upsample],
            dropout_prob=dropout
            ).to(device)
    else:
        model = locals() [model_name](4,3).to(device)

    return model

model = create_model(model_name) #doing it this wway might 

# if load_save==1:
    # model.load_state_dict(torch.load(load_path),strict=False)
    # print("loaded saved model ", load_path)