import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')

from Input.config import load_path,root_dir,workers
from Training.network import model
import pandas as pd
import numpy as np
import torch
from monai.data import DataLoader,decollate_batch
from Input.dataset import BratsDataset
from Input.localtransforms import test_transforms0,post_trans,train_transform,val_transform

def SegResNet_features(load_path, model=model):
    device = torch.device("cuda:0")
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    model.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(0)), strict=False)
    
    ds = BratsDataset(root_dir, transform=val_transform)
    mask_list=ds.mask_list
    print(len(set(mask_list)),len(mask_list))
    image_path=[x for x in mask_list][:1251]
    
    
    
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
    
    feature_df= pd.DataFrame(data=np.array(all_features),index=image_path)
    feature_df.index.name='mask_path'
    feature_df.to_csv('/scratch/a.bip5/BraTS/B23Encoded_maxfeatures.csv')
    
    return None

if __name__=='__main__':
    SegResNet_features(load_path, model=model)
    