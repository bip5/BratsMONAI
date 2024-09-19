import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')

import torch
import monai
import torch.nn.functional as F
import torch.nn as nn

class LinSig(torch.nn.Module):
    def __init__(self):
        super(LinSig, self).__init__()
        self.act_bias = nn.Parameter(torch.zeros(1))
        self.act_scale = nn.Parameter(torch.ones(1)) # Example parameters

    def forward(self, x):
        # Custom activation logic
        
        return self.act_scale * torch.relu(x + self.act_bias)

class LinRel(torch.nn.Module):
    def __init__(self):
        super(LinSig, self).__init__()
        self.params = torch.nn.Parameter(torch.randn(2))  # Example parameters

    def forward(self, x):
        # Custom activation logic
        linear_part = self.params[0] * x
        nonlinear_part = self.params[1] * torch.relu(x)
        return linear_part + nonlinear_part
class AllActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.3))
        self.b = torch.nn.Parameter(torch.tensor(0.3))
        self.c = torch.nn.Parameter(torch.tensor(0.3))

        self.register_parameter("a", self.a)
        self.register_parameter("b", self.b)
        self.register_parameter("c", self.c)

    def forward(self, x):
        return self.a * F.relu(x) + self.b * torch.sigmoid(x) + self.c * torch.tanh(x)


def get_act_layer(name):
    
    if name == 'LinSig':
        return LinSig()
    elif name == 'AllActivation':
        return AllActivation()
    elif name == 'hardswish':
        return torch.nn.Hardswish()
    else:        
        return monai.networks.layers.utils.get_act_layer (name)
