import monai
from Training.segresnetprj import SegResNet
import torch.nn as nn
from monai.utils import UpsampleMode
import torch
from Input.config import roi

class PixelLayer(SegResNet):
    def __init__(self,base_model,roi=roi,channels=4
        
    ):
        super(PixelLayer,self).__init__()
        self.base_model = base_model
        
        shape = (channels, *roi)       

        mean=1
        std=0.1
        self.roi=roi
        self.channels=channels
        self.alpha = nn.Parameter(torch.normal(mean, std, size=shape))
        self.init_activation=nn.Hardswish()
        
    def forward(self,x):
        x = x.float()              
        
        x = (self.alpha)*x 
        x=self.init_activation(x)
        
        return self.base_model(x)