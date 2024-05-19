import pandas
# print(pandas.__version__)
import nibabel

import os
import monai
from monai.data import Dataset
from monai.utils import set_determinism
from monai.apps import CrossValidation

# from monai.transforms import (EnsureChannelFirstD, AddChannelD,\
    # ScaleIntensityD, SpacingD, OrientationD,\
    # ResizeD, RandAffineD,
    # Activations,
    # Activationsd,
    # AsDiscrete,
    # AsDiscreted,
    # Compose,
    # Invertd,
    # LoadImaged,
    # RandBiasFieldD,
    # RandRotateD,
    # RotateD, Rotate,
    # RandGaussianSmoothD,
    # RandGaussianNoised,
    # MapTransform,
    # NormalizeIntensityd,
    # RandFlipd, RandFlip,
    # RandScaleIntensityd,
    # RandShiftIntensityd,
    # RandSpatialCropd,   
    # EnsureTyped,
    # EnsureType,
# )



from monai.losses import DiceLoss
from monai.utils import UpsampleMode
from monai.data import decollate_batch, list_data_collate

from monai.networks.nets import SegResNet, UNet
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




class ResBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs
        
class ecDoubleConv(nn.Module):
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
        x=nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)(x)
        
        return self.double_conv(x)
        
        
import torch
import torch.nn as nn
import torch.nn.functional as F

# class wDoubleConv(nn.Module):
    # """(convolution => [IN] => ReLU) * 2 with each convolution outputting one less channel and an additional sum channel, all normalized together"""

    # def __init__(self, in_channels, out_channels, mid_channels=None):
        # super().__init__()
        # if not mid_channels:
            # mid_channels = out_channels
        # self.conv1 = nn.Conv3d(in_channels, mid_channels - 1, kernel_size=3, padding=1, bias=False)
        # self.in1 = nn.InstanceNorm3d(mid_channels)  # Normalize all channels together
        
        # self.conv2 = nn.Conv3d(mid_channels, out_channels - 1, kernel_size=3, padding=1, bias=False)
        # self.in2 = nn.InstanceNorm3d(out_channels)  # Normalize all channels together

        # self.relu = nn.ReLU(inplace=True)

    # def forward(self, x):
        # # First convolution block
        # x = self.conv1(x)
        # sum_channel1 = torch.sum(x, dim=1, keepdim=True)
        # x = torch.cat([x, sum_channel1], dim=1)
        # x = self.in1(x)  # Instance normalize all channels together
        # x = self.relu(x)

        # # Second convolution block
        # x = self.conv2(x)
        # sum_channel2 = torch.sum(x, dim=1, keepdim=True)
        # x = torch.cat([x, sum_channel2], dim=1)
        # x = self.in2(x)  # Instance normalize all channels together
        # x = self.relu(x)

        # return x

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
            wDoubleConv(in_channels, out_channels)
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
            self.conv = wDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = wDoubleConv(in_channels, out_channels)

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
            self.conv = wDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = wDoubleConv(in_channels, out_channels)

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



    
        

class manUNet(nn.Module):
    def __init__(self, n_channels, n_classes,i=16, bilinear=False):
        super(manUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = wDoubleConv(n_channels,i )
        self.down1 = wDown(i, 2*i)
        self.down2 = wDown (2*i,2*2*i)
        self.down3 = wDown(2*2*i, 2*2*2*i)
        factor = 2 if bilinear else 1
        self.down4 = wDown(2*2*2*i, (16*i)// factor)
        self.up1 = wUp( (16*i),(8*i) // factor, bilinear)
        self.up2 = wUp( (8*i), (4*i) // factor, bilinear)
        self.up3 = wUp( 4*i, (2*i) // factor, bilinear)
        self.up4 = wUp( 2*i,i, bilinear)
        self.outc = wOutConv(i, n_classes)

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
    def __init__(self, n_channels, n_classes,i=8, bilinear=False):
        super(WNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = wDoubleConv(n_channels,i )
        self.down1 = wDown(i, 2*i)
        self.down2 = wDown (2*i,2*2*i)
        self.down3 = wDown(2*2*i, 2*2*2*i)
        factor = 2 if bilinear else 1
        self.down4 = wDown(2*2*2*i, (16*i)// factor)
        self.up01 = wUp0((16*i), 1 , bilinear)
        self.up02 = wUp0(2*2*2*i,1 , bilinear)
        self.up03 = wUp0(2*2*i, 1, bilinear)
        self.up04 = wUp0(2*i, 1, bilinear)
        self.outc = wOutConv(i, n_classes)
        self.up1 = wUp( (16*i),(8*i) // factor, bilinear)
        self.up2 = wUp( (8*i), (4*i) // factor, bilinear)
        self.up3 = wUp( 4*i, (2*i) // factor, bilinear)
        self.up4 = wUp( 2*i,i, bilinear)
        self.outc = wOutConv(i, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x04 = self.up01(x5)* x4
        x03 = self.up02(x04)* x3
        x02 = self.up03(x03)* x2
        x01 = self.up04(x02)* x1
        x12 = self.inc(x01)
        x22 = self.down1(x1)
        x32 = self.down2(x2)
        x42 = self.down3(x3)
        x52 = self.down4(x4)
        x004 = self.up1(x5, x42)
        x003 = self.up2(x04, x32)
        x002 = self.up3(x03,x22)
        x001 = self.up4(x02, x12)
        logits = self.outc(x001)
        return logits
        

        
class SegResNetAtt(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SegResNetAtt, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = nn.Sequential(
            nn.Conv3d(n_channels, 16, kernel_size=3, padding=1, bias=False,groups=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True))
        self.res1=DoubleConvAtt(16,16)
        
        self.down1 = DownConv(16,32,2)
        self.res2 = DoubleConvAtt(32, 32)
        
        self.down2 = DownConv(32, 64,2)
        self.res3 = DoubleConvAtt(64, 64)
        
        self.down3 = DownConv(64, 128,2)
        self.res4 = DoubleConvAtt(128, 128)
      
  
        
        self.up1 = sUp(128, 64, bilinear)
        self.up2 = sUp(64, 32 , bilinear)
        self.up3 = sUp(32, 16 , bilinear)
      
        self.outc = wOutConv(16, n_classes)

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

class LNDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,H,W,D, mid_channels=None):
        super(LNDoubleConv,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,groups=1),
            nn.LayerNorm([mid_channels,H,W,D],elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,groups=1),
            nn.LayerNorm([out_channels,H,W,D],elementwise_affine=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        if self.in_channels==self.out_channels:
            x=self.double_conv(x)+x
        else:
            x=self.double_conv(x)
        return x
        
class VANet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(VANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        filt=16
        self.pool1=DownConv(n_channels,filt,2)
        self.inc1=SpatialAttention(filt)
        
        self.pool2=DownConv(filt,filt*2,2)
        self.inc2=SpatialAttention(filt*2)
        
        self.pool3=DownConv(filt*2,filt*5,2)
        self.inc3=SpatialAttention(filt*5)
        
        self.pool4=DownConv(filt*5,filt*8,2)
        self.inc4=SpatialAttention(filt*8)
        
        self.up1 = UpConv(filt*8, filt*5)
        self.up2 = UpConv(filt*5, filt*2 )
        self.up3 = UpConv(filt*2, filt)
        self.up4 = UpConv(filt, int(filt/2))
               
        
        self.outc = OutConv(int(filt/2), n_classes)

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
        x=x+x3
        x = self.up2(x)
        # print("up2:", x.shape)
        x=x+x2
        x = self.up3(x)
        # print("up3:", x.shape)
        x=x+x1
        
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
        self.incfin = DoubleConv(24, 128)
        # self.incfin1 = DoubleConv(60, 60)
        
      
   
        self.up=UptoShape([192,192,144])
        self.pool1=DownConv(n_channels,n_channels,16)
        self.inc1 = DoubleConv(4, 512)#SpatialAttention(512)#DoubleConv(n_channels, 512)
        # self.inc11 = DoubleConv(512, 512)#SpatialAttention(512)# 
        self.squeeze1=OutConv(516,4)
        self.up1 = Up(16)
        
        self.pool2=DownConv(n_channels,n_channels,8)#AMpool(8)
        self.inc2=DoubleConv(4, 256)#SpatialAttention(256)#DoubleConv(n_channels*2, 256)
        # self.inc22 = DoubleConv(256, 256)#SpatialAttention(256)#DoubleConv(256, 256) 
        self.squeeze2=OutConv(260,4)
        self.up2 = Up(8)
        
        self.pool3=DownConv(n_channels,n_channels,4)
        self.inc3=DoubleConv(4, 128)#SpatialAttention(128)#DoubleConv(n_channels*2, 128)
        # self.inc33=DoubleConv(128, 128)#SpatialAttention(128)#
        self.squeeze3=OutConv(132,4)
        self.up3 = Up(4)
        
        self.pool4=DownConv(n_channels,n_channels,3)
        self.inc4=DoubleConv(4,96)#SpatialAttention(64)#
        # self.inc44=DoubleConv(64, 64)#SpatialAttention(64)#
        self.squeeze4=OutConv(100,4)
        self.up4=Up(3)
        
        
        self.pool5=DownConv(n_channels,n_channels,2)
        self.inc5=DoubleConv(4, 64)#SpatialAttention(64)#
        # self.inc44=DoubleConv(64, 64)#SpatialAttention(64)#
        self.squeeze5=OutConv(68,4)
        self.up5=Up(4)
        
        # self.pool6=DownConv(n_channels,n_channels,6)
        # self.inc6=DoubleConv(4, 4)#
        # self.squeeze6=OutConv(32,8)
        # self.up6=Up(12)
       
        

    def forward(self, x):      
       
        x1p = self.pool1(x)
        x1 = self.inc1(x1p)
        x1=torch.cat((x1p,x1),dim=1)        
        x1=self.squeeze1(x1)
        x1=self.up(x1)
        
        # x1=self.up(x1)
        # print(x1.shape)
        
        x2p = self.pool2(x)
        x2 = self.inc2(x2p)
        x2=torch.cat((x2p,x2),dim=1)
        x2=self.squeeze2(x2)
        x2=self.up(x2)
        
        
        # print(x2.shape)
        
        x3p = self.pool3(x)
        x3 = self.inc3(x3p)
        x3=torch.cat((x3p,x3),dim=1)
        x3=self.squeeze3(x3)
        x3=self.up(x3)
        
        
        # print(x3.shape)
        
        x4p = self.pool4(x)
        x4 = self.inc4(x4p)
        x4=torch.cat((x4p,x4),dim=1)
        x4=self.squeeze4(x4)
        x4=self.up(x4)
        
        
        # print(x4.shape)
        
        x5p = self.pool5(x)
        x5 = self.inc5(x5p)
        x5=torch.cat((x5p,x5),dim=1)
        x5=self.up(x5)
        x5=self.squeeze5(x5)

        
        xout=torch.cat((x,x1,x2,x3,x4,x5),dim=1)
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

class Residual(nn.Module):
    def __init__(self):
        super(Residual,self).__init__()
    def forward(self,x):
        return x
        
class EMul(nn.Module):
    def __init__(self):
        super(EMul,self).__init__()
    def forward(self,x):
        return x
        
class UptoShape(nn.Module):
    def __init__(self,size):
        super(UptoShape,self).__init__()
        self.up = nn.Upsample(size=size, mode='nearest')
    def forward(self,x):
        x=self.up(x)
        return x

class UpConv(nn.Module):
    def __init__(self,in_chan,out_chan):
        super(UpConv,self).__init__()
        self.up = nn.ConvTranspose3d(in_chan,out_chan,kernel_size=2,stride=2)        
        self.activation = nn.GELU()
        self.LN=nn.InstanceNorm3d(out_chan)
        self.conv=nn.Conv3d(out_chan,out_chan,kernel_size=3,padding=1)
        
        
    def forward(self,x):
        x=self.up(x)
        x=self.activation(x)
        x=self.LN(x)
        x=self.conv(x)
        x=self.activation(x)
        x=self.LN(x)
        
        return x  

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
            # nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,groups=1),
            nn.InstanceNorm3d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        if self.in_channels==self.out_channels:
            x=self.double_conv(x)+x
        else:
            x=self.double_conv(x)
        return x
        
class DoubleConvAtt(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, 2*mid_channels, kernel_size=3, padding=1, bias=False,groups=1),
            nn.InstanceNorm3d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv3d(2*mid_channels, 2*out_channels, kernel_size=3, padding=1, bias=False,groups=1),
            nn.InstanceNorm3d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        if self.in_channels==self.out_channels:
            out_channels=self.out_channels
            conv_output=self.double_conv(x)
            #Split the output into two halves
            split_size = out_channels
            first_half, second_half = torch.split(conv_output, split_size, dim=1)

           # Reshape for softmax application
            original_shape = first_half.shape
            first_half_flattened = first_half.view(original_shape[0], original_shape[1], -1)

            #Apply softmax across spatial dimensions
            first_half_softmax = torch.softmax(first_half_flattened, dim=2)

            #Reshape back to original dimensions
            first_half_softmax = first_half_softmax.view(original_shape)

            #Multiply with the second half
            conv_output = first_half_softmax * second_half
            
            x=conv_output+x
            
        else:
            x=self.double_conv(x)
        return x
        
class DownConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, factor, mid_channels=None):
        super(DownConv,self).__init__()
        
        
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=factor,stride=factor, padding=0, bias=False,groups=1),
            nn.BatchNorm3d(out_channels),
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

class LKConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel),
        nn.InstanceNorm3d(out_channels),
        nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv(x) 