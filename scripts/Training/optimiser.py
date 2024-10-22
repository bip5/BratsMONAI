import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')

import torch
import torch.nn as nn
from Training.network import model
from Input.config import lr,training_mode

optimiser = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
device = torch.device('cuda:0')

def get_optimiser(model):
    if training_mode =='isles':
        weights = torch.tensor([1.0, 0.5, 0.25,0.125], requires_grad=True).to(device) # example weights
        ds_wt=weights.detach().requires_grad_()
        optimiser = torch.optim.AdamW([{'params': model.parameters()}], lr, weight_decay=1e-5)  
    elif training_mode=='atlas':
        optimiser = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)  
    else:    
        optimiser =torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
    return optimiser