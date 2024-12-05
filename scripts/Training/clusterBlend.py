import monai
from Training.segresnetprj import SegResNet
import torch.nn as nn
from monai.utils import UpsampleMode
import torch

class ClusterBlend(SegResNet):
    def __init__(self,base_model,initial_alpha=0.1,
        
    ):
        super(ClusterBlend,self).__init__()
        self.base_model = base_model
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        
        
    def forward(self,x):
        '''
        Extract a cluster map and combine it with original.     
        `x[:, :4, :, :, :]` extracts the first 4 channels from the tensor.
        `x[:, 4:, :, :, :]` extracts the remaining channels from the tensor. 
        '''
        x = x.float()
    
        x1, x2 = x[:, :4, :, :, :], x[:, 4:, :, :, :]
        
        
        blended_x = (1-self.alpha)*x1 + self.alpha * x2
        
        return self.base_model(blended_x)