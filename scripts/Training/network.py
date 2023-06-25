import sys
sys.path.append('/scratch/a.bip5/BraTS 2021/scripts/')

from Input.config import (
model_name,
load_save,
seed,
load_path,
upsample,
)
import torch
from monai.networks.nets import UNet, SegResNet
from monai.utils import UpsampleMode


device = torch.device("cuda:0")
    
torch.manual_seed(seed)
if model_name=="UNet":
     model=UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(64,128,256,512,1024),
        strides=(2,2,2,2)
        ).to(device)
elif model_name=="SegResNet":
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
    model = locals() [model_name](4,3).to(device)


if load_save==1:
    model.load_state_dict(torch.load("/scratch/a.bip5/BraTS 2021/"+load_path),strict=False)
    print("loaded saved model ", load_path)