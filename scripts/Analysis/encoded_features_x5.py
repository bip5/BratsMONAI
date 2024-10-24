import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')

from Input.config import load_path,root_dir,workers,raw_features_filename
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


'''Script to extract features using a specified base model. This is the extended version where in addition to the max of each filter at the deepest encoder level, the min, the mean and variation will be extracted to return a 1024 element vector representing each sample'''

def model_loader(modelweight_path):
    ''' same function as in evaluation to open model for robustness'''
    from Training.network import create_model #should hopefully solve the issue
    # Create the model instance
    model = create_model()

    # Load the state dict from the file
    state_dict = torch.load(modelweight_path)

    # Check if the state dict contains keys prefixed with 'module.'
    # This indicates that the model was saved with DataParallel
    is_dataparallel = any(key.startswith('module.') for key in state_dict.keys())

    if is_dataparallel:
        # Wrap the model with DataParallel before loading the state dict
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict, strict=True)
    else:
        # If there's no 'module.' prefix, load the state dict as is
        # This also handles the case where the model needs to be wrapped but the saved model wasn't
        # If necessary, you can modify this part to adjust the keys in state_dict
        model.load_state_dict(state_dict, strict=True)
        
    return model

def SegResNet_features(load_path ):

    ds = BratsDataset(root_dir, transform=test_transforms1)
    # ds=Subset(full_ds,np.sort(train_indices))
    image_paths=[('/').join(x[0].split('/')[:-1]) for x in ds.image_list]
    # image_paths=[image_paths[x] for x in train_indices]
    
    
    model= model_loader(load_path)
    device = torch.device("cuda:0")

    model.to(device)   

    model.eval() 
    
    
    
    
    
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    
   
    
    
    list_features = [] # List to store features for all test images
    with torch.no_grad():
        for im_data in loader:  # Fixed variable name here
            all_features = [] #dict to store new info
            im_inputs = im_data["image"].to(device)
            if hasattr(model,'module'):
                encoded_maps, _ = model.module.encode(im_inputs)
            else:
                encoded_maps, _ = model.encode(im_inputs)
            mean_features = torch.mean(encoded_maps, dim=(2,3,4),keepdim=True)
            
            var_features = torch.var(encoded_maps, dim=(2, 3, 4), keepdim=True)

            # Reduce over the 4th dimension first
            max_val_4th_dim, _ = torch.max(encoded_maps, dim=4, keepdim=True)

            # Then reduce over the 3rd dimension
            max_val_3rd_dim, _ = torch.max(max_val_4th_dim, dim=3, keepdim=True)

            # Finally, reduce over the 2nd dimension
            max_features, _ = torch.max(max_val_3rd_dim, dim=2, keepdim=True)
            
            # Reduce over the 4th dimension first
            min_val_4th_dim, _ = torch.min(encoded_maps, dim=4, keepdim=True)

            # Then reduce over the 3rd dimension
            min_val_3rd_dim, _ = torch.min(min_val_4th_dim, dim=3, keepdim=True)

            # Finally, reduce over the 2nd dimension
            min_features, _ = torch.min(min_val_3rd_dim, dim=2, keepdim=True)
            
            all_features.extend(min_features.cpu().numpy().flatten())
            all_features.extend( max_features.cpu().numpy().flatten())
            all_features.extend( mean_features.cpu().numpy().flatten())
            all_features.extend( var_features.cpu().numpy().flatten())
            # Append features to the list
            list_features.append(all_features)
    
    feature_df= pd.DataFrame(data=np.array(list_features),index=image_paths[:-1])
    feature_df.index.name='mask_path'
    feature_df.to_csv('/scratch/a.bip5/BraTS/trainFeatures_5x.csv')
    
    return None
    
def single_encode_x5(test_inputs,model):
    all_features = []
    if hasattr(model,'module'):
        encoded_maps, _ = model.module.encode(test_inputs)
    else:
        encoded_maps, _ = model.encode(test_inputs)
    # features = torch.mean(encoded_maps, dim=(2,3,4),keepdim=True)
    mean_features = torch.mean(encoded_maps, dim=(2,3,4),keepdim=True)
            
    var_features = torch.var(encoded_maps, dim=(2, 3, 4), keepdim=True)
    # Reduce over the 4th dimension first
    max_val_4th_dim, _ = torch.max(encoded_maps, dim=4, keepdim=True)

    # Then reduce over the 3rd dimension
    max_val_3rd_dim, _ = torch.max(max_val_4th_dim, dim=3, keepdim=True)
    
    # Finally, reduce over the 2nd dimension
    max_features, _ = torch.max(max_val_3rd_dim, dim=2, keepdim=True)
    
    # Reduce over the 4th dimension first
    min_val_4th_dim, _ = torch.min(encoded_maps, dim=4, keepdim=True)

    # Then reduce over the 3rd dimension
    min_val_3rd_dim, _ = torch.min(min_val_4th_dim, dim=3, keepdim=True)

    # Finally, reduce over the 2nd dimension
    min_features, _ = torch.min(min_val_3rd_dim, dim=2, keepdim=True)

    
    all_features.extend(min_features.cpu().numpy().flatten())
    all_features.extend( max_features.cpu().numpy().flatten())
    all_features.extend( mean_features.cpu().numpy().flatten())
    all_features.extend( var_features.cpu().numpy().flatten())
    return np.array(all_features)


if __name__=='__main__':
    SegResNet_features(load_path)
    