import pandas
# print(pandas.__version__)
import nibabel
from typing import List, Optional, Sequence, Tuple, Union
import os
import monai
from monai.data import Dataset,partition_dataset
from monai.utils import set_determinism
from monai.apps import CrossValidation
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

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
    RandGaussianNoised,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd, RandFlip,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,   
    EnsureTyped,
    EnsureType,
)



from monai.losses import DiceLoss
from monai.utils import UpsampleMode
from monai.data import decollate_batch, list_data_collate

from monai.networks.nets import SegResNet,SegResNetVAE, UNet
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import DataLoader
import numpy as np
from datetime import date, datetime
import sys
import re
import torch
import torch.nn as nn
import time
import argparse
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import Subset
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
import matplotlib.pyplot as plt
import resource
from torch.utils.data.distributed import DistributedSampler
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["OMP_NUM_THREADS"] = "4"


        
if __name__=="__main__":

    parser=argparse.ArgumentParser(description="Monai Seg main")

    parser.add_argument("--lr",default=2e-4,type=float,help="learning rate")
    parser.add_argument("--model",default="SegResNetVAEx",type=str,help="name of model to use")
    parser.add_argument("--load_save",default =0, type=int,help="flag to use saved model weight")
    parser.add_argument("--load_path",default="/scratch/a.bip5/BraTS 2021/2022-01-20T16best_metric_model.pth", type=str, help="file path to load previously saved model")
    parser.add_argument("--batch_size",default=1, type=int, help="to define batch size")
    parser.add_argument("--save_name", default="SISANET.pth",type=str, help="save name")
    parser.add_argument("--upsample", default="DECONV",type=str, help="flag to choose deconv options- NONTRAINABLE, DECONV, PIXELSHUFFLE")
    parser.add_argument("--barlow_final",default=1, type=int, help="flag to use checkpoint instead of final model for barlow")
    parser.add_argument("--bar_model_name",default="checkpoint.pth", type=str,help="model name to load")
    parser.add_argument("--max_samples",default=10,type=int,help="max number of samples to use for training")
    parser.add_argument("--fold_num",default=1,type=str,help="cross-validation fold number")
    parser.add_argument("--epochs",default=1200,type=int,help="number of epochs to run")
    parser.add_argument("--CV_flag",default=1,type=int,help="is this a cross validation fold? 1=yes")
    parser.add_argument("--seed",default=0,type=int, help="random seed for the script")
    parser.add_argument("--method",default='A', type=str,help='A,B or C')
    parser.add_argument("--Tmax",default=4, type=int,help='number epochs to cycle lr')
    parser.add_argument("--workers",default=1, type=int,help='cpu cores used to load data')
    parser.add_argument("--DDP",default=0,type=int,help="to use data distributed parallel or not(not implemented yet)")
    args=parser.parse_args()

    print(' '.join(sys.argv))
    
    os.environ['PYTHONHASHSEED']=str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    set_determinism(seed=args.seed)

    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npy', '.gz'
    ]


    

    # A source: Nvidia HDGAN
    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    # makes a list of all image paths inside a directory
    def make_dataset(data_dir):
        all_files = []
        images=[]
        masks=[]
        im_temp=[]
        assert os.path.isdir(data_dir), '%s is not a valid directory' % data_dir
        
        for root, fol, _ in sorted(os.walk(data_dir)): # list folders and root
            for folder in fol:                    # for each folder
                 path=os.path.join(root, folder)  # combine root path with folder path
                 for root1, _, fnames in os.walk(path):       #list all file names in the folder         
                    for f in fnames:                          # go through each file name
                        fpath=os.path.join(root1,f)
                        if is_image_file(f):                  # check if expected extension
                            if re.search("seg",f):            # look for the mask files- have'seg' in the name 
                                masks.append(fpath)
                            else:
                                im_temp.append(fpath)         # all without seg are image files, store them in a list for each folder
                    images.append(im_temp)                    # add image files for each folder to a list
                    im_temp=[]
        return images, masks
        
    class SegResNetVAEdef(SegResNetVAE):
        def __init__(
            self,
            input_image_size: Sequence[int],
            vae_estimate_std: bool = False,
            vae_default_std: float = 0.3,
            vae_nz: int = 256,
            spatial_dims: int = 3,
            init_filters: int = 8,
            in_channels: int = 1,
            out_channels: int = 2,
            dropout_prob: Optional[float] = None,
            act: Union[str, tuple] = ("RELU", {"inplace": True}),
            norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
            use_conv_final: bool = True,
            blocks_down: tuple = (1, 2, 2, 4),
            blocks_up: tuple = (1, 1, 1),
            upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
        ):
            super().__init__(
                input_image_size=input_image_size,
                vae_estimate_std=vae_estimate_std,
                vae_default_std=vae_default_std, 
                vae_nz=vae_nz,                
                spatial_dims=spatial_dims,
                init_filters=init_filters,
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
                use_conv_final=use_conv_final,
                blocks_down=blocks_down,
                blocks_up=blocks_up,
                upsample_mode=upsample_mode,
            )
        def forward(self, x,mask=None):
            net_input = x
            x, down_x = self.encode(x)
            down_x.reverse()

            vae_input = x
            x = self.decode(x, down_x)

            if self.training:
               
               
                vae_loss = self._get_vae_loss(net_input, vae_input)
                return x, vae_loss

            return x

    class SegResNetVAEy(SegResNetVAE):
        def __init__(
            self,
            input_image_size: Sequence[int],
            vae_estimate_std: bool = False,
            vae_default_std: float = 0.3,
            vae_nz: int = 256,
            spatial_dims: int = 3,
            init_filters: int = 8,
            in_channels: int = 1,
            out_channels: int = 2,
            dropout_prob: Optional[float] = None,
            act: Union[str, tuple] = ("RELU", {"inplace": True}),
            norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
            use_conv_final: bool = True,
            blocks_down: tuple = (1, 2, 2, 4),
            blocks_up: tuple = (1, 1, 1),
            upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
        ):
            super().__init__(
                input_image_size=input_image_size,
                vae_estimate_std=vae_estimate_std,
                vae_default_std=vae_default_std, 
                vae_nz=vae_nz,                
                spatial_dims=spatial_dims,
                init_filters=init_filters,
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
                use_conv_final=use_conv_final,
                blocks_down=blocks_down,
                blocks_up=blocks_up,
                upsample_mode=upsample_mode,
            )
            
        def _get_vae_loss(self, net_input: torch.Tensor, vae_input: torch.Tensor,mask):
            """
            Args:
                net_input: the original input of the network.
                vae_input: the input of VAE module, which is also the output of the network's encoder.
            """
            mask=(mask>0).float()
            mask=torch.sum(mask,dim=1)
            # print(torch.unique(mask))
            mask=(mask==0).float()
            # print(mask.shape)
            mask=mask.unsqueeze(1)
            
            x_vae = self.vae_down(vae_input)
            x_vae = x_vae.view(-1, self.vae_fc1.in_features)
            z_mean = self.vae_fc1(x_vae)
            # print(net_input.shape)
            mask = mask.expand(-1, net_input.shape[1], -1, -1, -1)
           

            z_mean_rand = torch.randn_like(z_mean)
            z_mean_rand.requires_grad_(False)

            if self.vae_estimate_std:
                z_sigma = self.vae_fc2(x_vae)
                z_sigma = F.softplus(z_sigma)
                vae_reg_loss = 0.5 * torch.mean(z_mean**2 + z_sigma**2 - torch.log(1e-8 + z_sigma**2) - 1)

                x_vae = z_mean + z_sigma * z_mean_rand
            else:
                z_sigma = self.vae_default_std
                vae_reg_loss = torch.mean(z_mean**2)

                x_vae = z_mean + z_sigma * z_mean_rand

            x_vae = self.vae_fc3(x_vae)
            x_vae = self.act_mod(x_vae)
            x_vae = x_vae.view([-1, self.smallest_filters] + self.fc_insize)
            x_vae = self.vae_fc_up_sample(x_vae)

            for up, upl in zip(self.up_samples, self.up_layers):
                x_vae = up(x_vae)
                x_vae = upl(x_vae)

            x_vae = self.vae_conv_final(x_vae)
        
            vae_mse_loss = F.mse_loss(net_input*mask, x_vae*mask)
            vae_loss = vae_reg_loss + vae_mse_loss
            return vae_loss

        def forward(self, x,mask=None):
            net_input = x
            x, down_x = self.encode(x)
            down_x.reverse()

            vae_input = x
            x = self.decode(x, down_x)

            if self.training:
                assert mask is not None, "Mask required during training"
                vae_loss = self._get_vae_loss(net_input, vae_input,mask)
                return x, vae_loss

            return x

    class SegResNetVAEx(SegResNet):
      
        

        def __init__(
            self,
            input_image_size: Sequence[int],
            vae_estimate_std: bool = False,
            vae_default_std: float = 0.3,
            vae_nz: int = 256,
            spatial_dims: int = 3,
            init_filters: int = 8,
            in_channels: int = 1,
            out_channels: int = 2,
            dropout_prob: Optional[float] = None,
            act: Union[str, tuple] = ("RELU", {"inplace": True}),
            norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
            use_conv_final: bool = True,
            blocks_down: tuple = (1, 2, 2, 4),
            blocks_up: tuple = (1, 1, 1),
            upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
        ):
            super().__init__(
                spatial_dims=spatial_dims,
                init_filters=init_filters,
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
                use_conv_final=use_conv_final,
                blocks_down=blocks_down,
                blocks_up=blocks_up,
                upsample_mode=upsample_mode,
            )

            self.input_image_size = input_image_size
            self.smallest_filters = 16

            zoom = 2 ** (len(self.blocks_down) - 1)
            self.fc_insize = [s // (2 * zoom) for s in self.input_image_size]

            self.vae_estimate_std = vae_estimate_std
            self.vae_default_std = vae_default_std
            self.vae_nz = vae_nz
            self._prepare_vae_modules()
            self.vae_conv_final = self._make_final_conv(in_channels)

        def _prepare_vae_modules(self):
            zoom = 2 ** (len(self.blocks_down) - 1)
            v_filters = self.init_filters * zoom
            total_elements = int(self.smallest_filters * np.prod(self.fc_insize))

            self.vae_down = nn.Sequential(
                get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
                self.act_mod,
                get_conv_layer(self.spatial_dims, v_filters, self.smallest_filters, stride=2, bias=True),
                get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.smallest_filters),
                self.act_mod,
            )
            self.vae_fc1 = nn.Linear(total_elements, self.vae_nz)
            self.vae_fc2 = nn.Linear(total_elements, self.vae_nz)
            self.vae_fc3 = nn.Linear(self.vae_nz, total_elements)

            self.vae_fc_up_sample = nn.Sequential(
                get_conv_layer(self.spatial_dims, self.smallest_filters, v_filters, kernel_size=1),
                get_upsample_layer(self.spatial_dims, v_filters, upsample_mode=self.upsample_mode),
                get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
                self.act_mod,
            )

        def _get_vae_loss(self, net_input: torch.Tensor, vae_input: torch.Tensor,mask):
            """
            Args:
                net_input: the original input of the network.
                vae_input: the input of VAE module, which is also the output of the network's encoder.
            """
            mask=(mask>0).float()
            mask=torch.sum(mask,dim=1)
            # print(torch.unique(mask))
            
            # print(mask.shape)
            mask=mask.unsqueeze(1)
            
            x_vae = self.vae_down(vae_input)
            x_vae = x_vae.view(-1, self.vae_fc1.in_features)
            z_mean = self.vae_fc1(x_vae)
            # print(net_input.shape)
            mask = mask.expand(-1, net_input.shape[1], -1, -1, -1)
           

            z_mean_rand = torch.randn_like(z_mean)
            z_mean_rand.requires_grad_(False)

            if self.vae_estimate_std:
                z_sigma = self.vae_fc2(x_vae)
                z_sigma = F.softplus(z_sigma)
                vae_reg_loss = 0.5 * torch.mean(z_mean**2 + z_sigma**2 - torch.log(1e-8 + z_sigma**2) - 1)

                x_vae = z_mean + z_sigma * z_mean_rand
            else:
                z_sigma = self.vae_default_std
                vae_reg_loss = torch.mean(z_mean**2)

                x_vae = z_mean + z_sigma * z_mean_rand

            x_vae = self.vae_fc3(x_vae)
            x_vae = self.act_mod(x_vae)
            x_vae = x_vae.view([-1, self.smallest_filters] + self.fc_insize)
            x_vae = self.vae_fc_up_sample(x_vae)

            for up, upl in zip(self.up_samples, self.up_layers):
                x_vae = up(x_vae)
                x_vae = upl(x_vae)

            x_vae = self.vae_conv_final(x_vae)
        
            vae_mse_loss = F.mse_loss(net_input*(1-mask), x_vae)
            vae_loss = vae_reg_loss + vae_mse_loss
            return vae_loss

        def forward(self, x,mask=None):
            net_input = x
            x, down_x = self.encode(x)
            down_x.reverse()

            vae_input = x
            x = self.decode(x, down_x)

            if self.training:
                assert mask is not None, "Mask required during training"
                vae_loss = self._get_vae_loss(net_input, vae_input,mask)
                return x, vae_loss

            return x
    # A source: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
    class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
        """
        Convert masks to multi channels based on brats classes:
        mask 2 is the peritumoral edema
        mask 4 is the GD-enhancing tumor
        mask 1 is the necrotic and non-enhancing tumor core
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor).

        """

        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                result = []
                # merge for TC
                result.append(np.logical_or(d[key] == 1, d[key] == 4))
                # merge masks 1, 2 and 3 to construct WT
                result.append(
                    np.logical_or(
                        np.logical_or(d[key] == 2, d[key] == 4), d[key] == 1
                    )
                )
                # mask ET
                result.append(d[key] == 4)
                d[key] = np.stack(result, axis=0).astype(np.float32)
            return d


    indexes=np.arange(args.max_samples)
    fold=int(args.max_samples/5)

    for i in range(1,6):
        if i==int(args.fold_num):
            if i<5:
                val_indices=indexes[(i-1)*fold:i*fold]
                train_indices=indexes[i*fold:(i+1)*fold]#np.delete(indexes,val_indices)#
            else:
                val_indices=indexes[(i-1)*fold:i*fold]
                train_indices=indexes[(i-5)*fold:(i-4)*fold]#np.delete(indexes,val_indices)#
                
               
    class BratsDataset(Dataset):
        def __init__(self,data_dir,transform=None):
            
            self.image_list=make_dataset(data_dir)[0]
             
            self.mask_list=make_dataset(data_dir)[1] 
            self.transform=transform
            
        def __len__(self):
    #         return len(os.listdir(self.mask_dir))
            return min(args.max_samples,len(self.mask_list))#
        
        def __getitem__(self,idx):
            # print(idx)
           
            image=self.image_list[idx]
        
            mask=self.mask_list[idx] 

                
            item_dict={"image":image,"mask":mask}
            
            if self.transform:
                item_dict={"image":image,"mask": mask}
                item_dict=self.transform(item_dict)
                
            
            return item_dict




        
    KEYS=("image","mask")
    print("Transforms not defined yet")
    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstD(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
            SpacingD(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            OrientationD(keys=["image", "mask"], axcodes="RAS"),
            RandSpatialCropd(keys=["image", "mask"], roi_size=[128, 128, 128], random_size=False),
           
            RandRotateD(keys=["image","mask"],range_x=0.1,range_y=0.1, range_z=0.1,prob=0.5),
           
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            EnsureTyped(keys=["image", "mask"]),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstD(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
            SpacingD(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            OrientationD(keys=["image", "mask"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandSpatialCropd(keys=["image", "mask"], roi_size=[128, 128, 128], random_size=False),
            EnsureTyped(keys=["image", "mask"]),
        ]
    )

    #dataset=DecathlonDataset(root_dir="./", task="Task05_Prostate",section="training", transform=xform, download=True)
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend="nccl", init_method="env://")
    train_dataset=partition_dataset(data=BratsDataset( "/scratch/a.bip5/BraTS 2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData"  ,transform=train_transform ), shuffle=False,num_partitions=dist.get_world_size(),even_divisible=False)[dist.get_rank()] 
    val_dataset=partition_dataset(data=BratsDataset( "/scratch/a.bip5/BraTS 2021/RSNA_ASNR_MICCAI_BraTS2021_TestData"  ,transform=val_transform ), shuffle=False,num_partitions=dist.get_world_size(),even_divisible=False)[dist.get_rank()]

    


    # if args.CV_flag==1:
        # print("loading cross val data")
        # val_dataset=Subset(train_dataset,val_indices)
        # train_dataset=Subset(train_dataset,train_indices)
        
    # else:     
        # print("loading data for single model training")
        # start=int(args.max_samples//10*8)
        # val_dataset=Subset(train_dataset,np.arange(start,args.max_samples))
        # train_dataset=Subset(train_dataset,np.arange(start))
        
        
    print("number of files processed: ", train_dataset.__len__())
    train_sampler = DistributedSampler(train_dataset)
    train_loader=DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,num_workers=args.workers)
    val_sampler = DistributedSampler(val_dataset)
    val_loader=DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,num_workers=args.workers)
    print("All Datasets assigned")

    root_dir="/scratch/a.bip5/BraTS 2021/"

    max_epochs = args.epochs
    val_interval = 1
    VAL_AMP = True

    # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    
    torch.manual_seed(args.seed)
    if args.model=="UNet":
         model=UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(64,128,256,512,1024),
            strides=(2,2,2,2)
            ).to(device)
    elif args.model=="SegResNetVAEx":
        model = SegResNetVAEx(
        input_image_size=(128,128,128),
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=32,
            norm="instance",
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[args.upsample]    
            ).to(device)
    elif args.model=="SegResNetVAE":
        model = SegResNetVAEdef(
        input_image_size=(128,128,128),
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=32,
            norm="instance",
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[args.upsample]    
            ).to(device)
    elif args.model=="SegResNet":
        model = SegResNet(       
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=32,
            norm="instance",
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[args.upsample]    
            ).to(device)

    else:
        model = locals() [args.model](4,3).to(device)

    # with torch.cuda.amp.autocast():
        # summary(model,(4,128,128,128),(128,128,128))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    model=model.to(device)
    
    model = DistributedDataParallel(module=model, device_ids=[device],find_unused_parameters=True)
    #model=torch.nn.DataParallel(model)
    print("Model defined and passed to GPU")
    
    if args.load_save==1:
        model.load_state_dict(torch.load("/scratch/a.bip5/BraTS 2021/saved models/"+args.load_path),strict=False)
        print("loaded saved model ", args.load_path)

    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Tmax)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    post_trans = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        
    def inference(input):

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(128,128, 128),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
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

    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    total_start = time.time()

    print("starting epochs")
    losses=[]
    lr_list=[]
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, masks = (
                batch_data["image"].to(device),
                batch_data["mask"].to(device),
            )
            optimizer.zero_grad()
            if args.model=='SegResNetVAEx':         
                with torch.cuda.amp.autocast():
                    outputs,vae_loss = model(inputs,masks)
                    loss = loss_function(outputs, masks)
                    total_loss=loss+vae_loss
                scaler.scale(total_loss).backward()
            elif args.model=='SegResNetVAE':
                with torch.cuda.amp.autocast():
                    outputs,vae_loss= model(inputs)
                    loss = loss_function(outputs,masks)
                    total_loss=loss+vae_loss
                scaler.scale(total_loss).backward()
                
            else:
                with torch.cuda.amp.autocast():
                    outputs= model(inputs)
                    loss = loss_function(outputs,masks)                    
                scaler.scale(loss).backward()
                
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            if step%10==0:
                print(
                    f"{step}/{len(train_dataset) // train_loader.batch_size}"
                    f", train_loss: {loss.item():.4f}"
                    f", step time: {(time.time() - step_start):.4f}"
                )
        
        print("lr_scheduler.get_last_lr() = ",lr_scheduler.get_last_lr())
        lr_list.append(lr_scheduler.get_last_lr())
        if epoch>99:
            lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        losses.append(epoch_loss)
        # if epoch>50:
            # torch.save(
                            # model.state_dict(),
                            # os.path.join(root_dir, date.today().isoformat()+'T'+str(datetime.today().hour)+ args.model+"ep"+str(epoch+1)))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():

                for val_data in val_loader:
                    val_inputs, val_masks = (
                        val_data["image"].to(device),
                        val_data["mask"].to(device),
                    )
                    
                    val_outputs = inference(val_inputs)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    
                    dice_metric(y_pred=val_outputs, y=val_masks)
                    dice_metric_batch(y_pred=val_outputs, y=val_masks)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                metric_tc = metric_batch[0].item()
                metric_values_tc.append(metric_tc)
                metric_wt = metric_batch[1].item()
                metric_values_wt.append(metric_wt)
                metric_et = metric_batch[2].item()
                metric_values_et.append(metric_et)
                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    if args.CV_flag==1:
                        torch.save(
                            model.state_dict(),
                            os.path.join(root_dir, args.model+"CV"+str(args.fold_num)+"ms"+str(args.max_samples)+"rs"+str(args.seed)+args.method)
                        )
                    else:
                        torch.save(
                            model.state_dict(),
                            os.path.join(root_dir, date.today().isoformat()+'T'+str(datetime.today().hour)+ args.model),
                        )
                    print("saved new best metric model")
                    
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
        

                
        print(f"time consumption of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    plt.figure()
    fig, ax1 = plt.subplots()
    

    # Plot the epoch losses on the primary y-axis
    ax1.plot(range(1, epoch + 2), losses, label='Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis and plot the best_metric on it
    ax2 = ax1.twinx()
    ax2.plot(range(1, epoch + 2), lr_list, label='Lr', color='green')
    ax2.set_ylabel('Leraning Rate', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Set the title and save the figure
    plt.title('Training Loss Progress and Learning Rate')
    if args.DDP:
        fig.savefig('loss_progress_DDP.png', dpi=300) 
    else:
        fig.savefig('loss_progress_single.png', dpi=300)
    total_time = time.time() - total_start

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
    with open ('/scratch/a.bip5/BraTS 2021/time_consumption.csv', 'a') as sample:
        sample.write(f"{args.model},{args.method},{total_time},{date.today().isoformat()},{args.fold_num},{args.CV_flag},{args.seed},,{args.epochs}\n")
        print("File write sucessful")
