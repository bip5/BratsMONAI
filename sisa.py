import pandas
# print(pandas.__version__)
import nibabel

import os
import monai
from monai.data import Dataset
from monai.utils import set_determinism
from monai.apps import CrossValidation

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

from monai.networks.nets import SegResNet
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

parser=argparse.ArgumentParser(description="Monai Seg main")

parser.add_argument("--lr",default=1e-3,type=float,help="learning rate")
parser.add_argument("--model",default="SISANet",type=str,help="name of model to use")
parser.add_argument("--load_save",default =1, type=int,help="flag to use saved model weight")
parser.add_argument("--load_path",default="./2022-01-20T16best_metric_model.pth", type=str, help="file path to load previously saved model")
parser.add_argument("--batch_size",default=2, type=int, help="to define batch size")
parser.add_argument("--save_name", default="SISANET.pth",type=str, help="save name")
parser.add_argument("--upsample", default="DECONV",type=str, help="flag to choose deconv options- NONTRAINABLE, DECONV, PIXELSHUFFLE")
parser.add_argument("--barlow_final",default=1, type=int, help="flag to use checkpoint instead of final model for barlow")
parser.add_argument("--bar_model_name",default="checkpoint.pth", type=str,help="model name to load")
parser.add_argument("--max_samples",default=10000,type=int,help="max number of samples to use for training")

args=parser.parse_args()

print(' '.join(sys.argv))

set_determinism(seed=0)
torch.manual_seed(0)
np.random.seed(0)
os.environ['PYTHONHASHSEED']=str(0)

class ResBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

class wDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class wDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class wUp0(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        # input is CHW
       
        return self.conv(x1)
        
class wUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class sUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_chan, out_chan, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.sqconv=nn.Conv3d(in_chan,out_chan,kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            # self.conv = DoubleConv(out_chan, out_chan)


    def forward(self, x):
        x1=self.sqconv(x)
        x2 = self.up(x1)
        
        # input is CHW
        
        return x2#self.conv(x2)

class wOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(wOutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




        

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = wDoubleConv(n_channels, 64)
        self.down1 = wDown(64, 128)
        self.down2 = wDown(128, 256)
        self.down3 = wDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = wDown(512, 1024 // factor)
        self.up1 = wUp(1024, 512 // factor, bilinear)
        self.up2 = wUp(512, 256 // factor, bilinear)
        self.up3 = wUp(256, 128 // factor, bilinear)
        self.up4 = wUp(128, 64, bilinear)
        self.outc = wOutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        
class WNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = wDoubleConv(n_channels, 64)
        self.down1 = wDown(64, 128)
        self.down2 = wDown(128, 256)
        self.down3 = wDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = wDown(512, 1024 // factor)
        self.up01 = wUp(1024, 1 , bilinear)
        self.up02 = wUp(512,1 , bilinear)
        self.up03 = wUp(256, 1, bilinear)
        self.up04 = wUp(128, 1, bilinear)
        self.outc = wOutConv(64, n_classes)
        self.up1 = wUp(1024, 512 // factor, bilinear)
        self.up2 = wUp(512, 256 // factor, bilinear)
        self.up3 = wUp(256, 128 // factor, bilinear)
        self.up4 = wUp(128, 64, bilinear)
        self.outc = wOutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x04 = self.up01(x5)* x4
        x03 = self.up02(x04)* x3
        x02 = self.up03(x03)*x2
        x01 = self.up04(x02)* x1
        x12 = self.inc(x01)
        x22 = self.down1(x1)
        x32 = self.down2(x2)
        x42 = self.down3(x3)
        x52 = self.down4(x4)
        logits = self.outc(x)
        return logits
        

        
class SegResNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SegResNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = nn.Sequential(
            nn.Conv3d(n_channels, 32, kernel_size=3, padding=1, bias=False,groups=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True))
        self.res1=DoubleConv(32,32)
        
        self.down1 = DownConv(32,64,2)
        self.res2 = DoubleConv(64, 64)
        
        self.down2 = DownConv(64, 128,2)
        self.res3 = DoubleConv(128, 128)
        
        self.down3 = DownConv(128, 256,2)
        self.res4 = DoubleConv(256, 256)
      
  
        
        self.up1 = sUp(256, 128, bilinear)
        self.up2 = sUp(128, 64 , bilinear)
        self.up3 = sUp(64, 32 , bilinear)
      
        self.outc = wOutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1=self.res1(x1)  
        
        x2 = self.down1(x1)
        x2=self.res2(x2)
        x2=self.res2(x2)
        
        x3 = self.down2(x2)
        x3=self.res3(x3)
        x3=self.res3(x3)
        
        x4 = self.down3(x3)
        x4=self.res4(x4)
        x4=self.res4(x4)
        x4=self.res4(x4)
        x4=self.res4(x4)
        
        x3up = self.up1(x4)
        x3up=x3+x3up
        
        x2up = self.up2(x3up)
        x2up=x2+x2up
        
        x1up = self.up3(x2up)
        x1up=x1+x1up
        
        logits = self.outc(x1up)
        return logits
        
class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)
        self.LN=nn.InstanceNorm3d(d_model)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

        
class VANet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(VANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        
        self.pool1=DownConv(n_channels,64,2)
        self.inc1=SpatialAttention(64)
        
        self.pool2=DownConv(64,128,2)
        self.inc2=SpatialAttention(128)
        
        self.pool3=DownConv(128,320,2)
        self.inc3=SpatialAttention(320)
        
        self.pool4=DownConv(320,512,2)
        self.inc4=SpatialAttention(512)
        
        self.up1 = sUp(512, 256, bilinear)
        self.up2 = sUp(256, 128 , bilinear)
        self.up3 = sUp(128, 64 , bilinear)
        self.up4 = sUp(64, 32 , bilinear)
               
        
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.pool1(x)
        x1=self.inc1(x1)
        x1=self.inc1(x1)
        x1=self.inc1(x1)
        # print("X1", x1.shape)
        
        x2 = self.pool2(x1)
        x2=self.inc2(x2)
        x2=self.inc2(x2)
        x2=self.inc2(x2)
        # print("X2:", x2.shape)
        
        x3 = self.pool3(x2)
        x3=self.inc3(x3)
        x3=self.inc3(x3)
        x3=self.inc3(x3)
        x3=self.inc3(x3)
        x3=self.inc3(x3)
        # print("X3:", x3.shape)
        
        x4 = self.pool4(x3)
        x4=self.inc4(x4)
        x4=self.inc4(x4)
        # print("X4:", x4.shape)
               
        
        
        x = self.up1(x4)
        # print("up1:", x.shape)
        x = self.up2(x)
        # print("up2:", x.shape)
        x = self.up3(x)
        # print("up3:", x.shape)
        x = self.up4(x)
        # print("up4:", x.shape)
        
        
        logits = self.outc(x)
        
        return logits
        
class SISANet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SISANet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.outc = OutConv(128, n_classes)
        self.incfin = DoubleConv(20, 128)
        # self.incfin1 = DoubleConv(60, 60)
        
      
   
     
        self.pool1=DownConv(n_channels,n_channels,16)
        self.inc1 = DoubleConv(4, 512)#SpatialAttention(512)#DoubleConv(n_channels, 512)
        # self.inc11 = DoubleConv(512, 512)#SpatialAttention(512)# 
        self.squeeze1=OutConv(512,4)
        self.up1 = Up(16)
        
        self.pool2=DownConv(n_channels,n_channels,8)#AMpool(8)
        self.inc2=DoubleConv(4, 256)#SpatialAttention(256)#DoubleConv(n_channels*2, 256)
        # self.inc22 = DoubleConv(256, 256)#SpatialAttention(256)#DoubleConv(256, 256) 
        self.squeeze2=OutConv(256,4)
        self.up2 = Up(8)
        
        self.pool3=DownConv(n_channels,n_channels,4)
        self.inc3=DoubleConv(4, 128)#SpatialAttention(128)#DoubleConv(n_channels*2, 128)
        # self.inc33=DoubleConv(128, 128)#SpatialAttention(128)#
        self.squeeze3=OutConv(128,4)
        self.up3 = Up(4)
        
        self.pool4=DownConv(n_channels,n_channels,2)
        self.inc4=DoubleConv(4, 64)#SpatialAttention(64)#
        # self.inc44=DoubleConv(64, 64)#SpatialAttention(64)#
        self.squeeze4=OutConv(64,4)
        self.up=Up(2)
        
        # self.pool5=DownConv(n_channels,64,4)
        # self.inc5=DoubleConv(64, 64)#SpatialAttention(64)#
        # self.inc44=DoubleConv(64, 64)#SpatialAttention(64)#
        # self.squeeze5=OutConv(64,8)
        # self.up5=Up(4)
        
        # self.pool6=DownConv(n_channels,32,2)
        # self.inc6=DoubleConv(32, 32)#
        # self.squeeze6=OutConv(32,8)
        # self.up6=Up(2)
       
        

    def forward(self, x):      
       
        x1p = self.pool1(x)
        x1 = self.inc1(x1p)
        x1=self.up(x1)
        x1=self.squeeze1(x1)
        # x1=self.up(x1)
        # print(x1.shape)
        
        x2p = self.pool2(x)
        x2 = self.inc2(x2p)
        x2=self.up(x2)
        x2=self.squeeze2(x2)
        # print(x2.shape)
        
        x3p = self.pool3(x)
        x3 = self.inc3(x3p)
        x3=self.up(x3)
        x3=self.squeeze3(x3)
        # print(x3.shape)
        
        x4p = self.pool4(x)
        x4 = self.inc4(x4p)
        x4=self.up(x4)
        x4=self.squeeze4(x4)
        # print(x4.shape)
        
        # x5p = self.pool5(x)
        # x5 = self.inc5(x5p)
        # x5=self.up5(x5)
        # x5=self.squeeze5(x5)
        # print(x5.shape)
        
        # x6p = self.pool6(x)
        # x6 = self.inc6(x6p)
        # x6=self.up6(x6)
        # x6=self.squeeze6(x6)
        # print(x6.shape)
               
                
        
        # x2p = self.pool2(x) 
        # x2=self.inc2(x2p)
    
        
        # x2=self.up(x2)
        
        
        # x3p=self.pool3(x)
 
        # x3=self.inc3(x3p)
        # x3=self.inc33(x3)
   
        # x3=self.up(x3)   
        
        # x4p = self.pool4(x) 
   
        # x4=self.inc4(x4p)
        # x4=self.inc44(x4)

        
        
        # x4=self.up(x4)
        x1_fin=self.up(self.up(self.up(x1)))
        x2_fin=self.up(self.up(x2))
        x3_fin=self.up(x3)
        
        xout = torch.cat((x4,x),dim=1)
        xout = torch.cat((x4,x1_fin),dim=1)
        xout=torch.cat((xout,x2_fin),dim=1)
        xout=torch.cat((xout,x3_fin),dim=1)
        xout=torch.cat((xout,x),dim=1)
        
        # xout=torch.cat((x,x1,x2,x3,x4),dim=1)
        xout = self.incfin(xout)
        # xout = self.incfin1(xout)
        logits = self.outc(xout)
        
        return logits

class AMpool(nn.Module):
    def __init__(self,factor):
        super(AMpool,self).__init__()
        self.factor=factor
        
        
    def forward(self,x):
        dim_x=int(x.shape[-3]/self.factor)
        dim_y=int(x.shape[-2]/self.factor)
        dim_z=int(x.shape[-1]/self.factor)
        dims=(dim_x,dim_y,dim_z)  #This is def a tuple of ints not a list despite what the error might say
        # print(type(dims), "dims type")
        
        
        # x1=nn.AdaptiveAvgPool3d(dims)(x)
        x2=nn.AdaptiveMaxPool3d(dims)(x)
        # x=torch.cat((x1,x2),dim=1)
        
        return x2

class Up(nn.Module):
    def __init__(self,factor):
        super(Up,self).__init__()
        self.up = nn.Upsample(scale_factor=factor, mode='nearest')
       
    
    def forward(self,y):
        # dim_x=y.shape[-3]*2
        # dim_y=y.shape[-2]*2
        # dim_z=y.shape[-1]*2        
        # y=nn.AdaptiveMaxPool3d((dim_x,dim_y,dim_z))(y)
        y=self.up(y)
        
        return y

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,groups=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,groups=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        # if self.in_channels==self.out_channels:
            # x=self.double_conv(x)+x
        # else:
        x=self.double_conv(x)
        return x
        
class DownConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, factor, mid_channels=None):
        super(DownConv,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=factor,stride=factor, padding=0, bias=False,groups=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        
        )
    def forward(self, x):
        return self.down_conv(x)  

        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)  


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npy', '.gz'
]


set_determinism(seed=0)

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
    


# A source: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert masks to multi channels based on brats classes:
    mask 1 is the peritumoral edema
    mask 2 is the GD-enhancing tumor
    mask 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge mask 2 and mask 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge masks 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # mask 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d
            
class BratsDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        self.image_list=make_dataset(data_dir)[0]  
        self.mask_list=make_dataset(data_dir)[1] 
        self.transform=transform
        
    def __len__(self):
#         return len(os.listdir(self.mask_dir))
        return min(args.max_samples,len(self.mask_list))
    
    def __getitem__(self,idx):
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
        RandSpatialCropd(keys=["image", "mask"], roi_size=[192, 192, 144], random_size=False),
       
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
        RandSpatialCropd(keys=["image", "mask"], roi_size=[192, 192, 144], random_size=False),
        EnsureTyped(keys=["image", "mask"]),
    ]
)

#dataset=DecathlonDataset(root_dir="./", task="Task05_Prostate",section="training", transform=xform, download=True)
train_dataset=BratsDataset("./RSNA_ASNR_MICCAI_BraTS2021_TrainingData"  ,transform=train_transform )
val_dataset=BratsDataset("./RSNA_ASNR_MICCAI_BraTS2021_ValidationData",transform=val_transform)
train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
print("All Datasets assigned")

root_dir="./"

max_epochs = 100
val_interval = 1
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = locals() [args.model](4,3).to(device)

with torch.cuda.amp.autocast():
    summary(model,(4,192,192,144))

model=torch.nn.DataParallel(model)
print("Model defined and passed to GPU")

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
def inference(input):

    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(192,192, 144),
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
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, masks)
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
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

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
                torch.save(
                    model.state_dict(),
                    os.path.join(root_dir, date.today().isoformat()+'T'+str(datetime.today().hour)+ args.save_name),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consumption of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

