import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')

from Input.config import load_path,root_dir,workers
from Training.network import model
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Subset
from monai.data import DataLoader,decollate_batch
from Input.dataset import (
test_indices,
train_indices,
val_indices
)
from Input.dataset import Brats23valDataset,BratsDataset
from Input.localtransforms import test_transforms0,post_trans,train_transform,val_transform,test_transforms1

def SegResNet_features(load_path, model=model ):

    ds = BratsDataset(root_dir, transform=test_transforms1)
    # ds=Subset(full_ds,np.sort(train_indices))
    image_paths=[('/').join(x[0].split('/')[:-1]) for x in ds.image_list]
    # image_paths=[image_paths[x] for x in train_indices]
    
    
    device = torch.device("cuda:0")
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    model.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(0)), strict=False)
    
    
    
    
    
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    
    model.eval()
    
    all_features = []  # List to store features for all test images
    
    with torch.no_grad():
        for test_data in loader:  # Fixed variable name here
            test_inputs = test_data["image"].to(device)
            encoded_maps, _ = model.module.encode(test_inputs)
            # features = torch.mean(encoded_maps, dim=(2,3,4),keepdim=True)
            # Reduce over the 4th dimension first
            max_val_4th_dim, _ = torch.max(encoded_maps, dim=4, keepdim=True)

            # Then reduce over the 3rd dimension
            max_val_3rd_dim, _ = torch.max(max_val_4th_dim, dim=3, keepdim=True)

            # Finally, reduce over the 2nd dimension
            features, _ = torch.max(max_val_3rd_dim, dim=2, keepdim=True)

            
            # Append features to the list
            all_features.append(features.cpu().numpy().flatten())
    
    feature_df= pd.DataFrame(data=np.array(all_features),index=image_paths[:-1])
    feature_df.index.name='mask_path'
    feature_df.to_csv('/scratch/a.bip5/BraTS/trainFeatures_17thJan.csv')
    
    return None
    
def single_encode(test_inputs,model):
    encoded_maps,_=model.module.encode(test_inputs)
    # features = torch.mean(encoded_maps, dim=(2,3,4),keepdim=True)
    # Reduce over the 4th dimension first
    max_val_4th_dim, _ = torch.max(encoded_maps, dim=4, keepdim=True)

    # Then reduce over the 3rd dimension
    max_val_3rd_dim, _ = torch.max(max_val_4th_dim, dim=3, keepdim=True)

    # Finally, reduce over the 2nd dimension
    features, _ = torch.max(max_val_3rd_dim, dim=2, keepdim=True)
    return features.cpu().numpy().flatten()


if __name__=='__main__':
    SegResNet_features(load_path, model=model)
    