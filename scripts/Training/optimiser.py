import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')

import torch
import torch.nn as nn
from network import model
from Input.config import lr

optimiser = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)