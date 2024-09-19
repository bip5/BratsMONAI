import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')

from Input.config import load_path,root_dir,workers, model_name,roi
from Training.network import model
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Subset
from monai.data import DataLoader,decollate_batch
from Input.dataset import (
test_indices,
train_indices,
val_indices,
temporal_split,

)
import copy
from Input.dataset import Brats23valDataset,BratsDataset
from Input.localtransforms import test_transforms0,post_trans,train_transform,val_transform,test_transforms1,transformer_transform
from datetime import datetime
from Analysis.swin_encoder import SwinUNETREncoder

from monai.transforms import CenterSpatialCrop


'''Script to extract features using a specified base model'''



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
    

# Hook function definition with debug prints
def get_activation(sample_id, activation):
    def hook(model, input, output):
        # print(f"Hook called for sample_id: {sample_id}")
        
        # print("input max value:", input[0].max().item())
        # print("Output shape:", output.shape)
        # print("Output max value:", output.max().item())
        # print("Output min value:", output.min().item())
        
        # Squeeze and process the output
        squeezed_output = torch.squeeze(torch.max(torch.max(torch.max(output, dim=4)[0], dim=3)[0], dim=2)[0]).detach().cpu().numpy()
        
      
        # print('Squeezed output shape:', squeezed_output.shape)
        # print('Squeezed output max value:', squeezed_output.max())
        # print('Squeezed output min value:', squeezed_output.min())
        # sys.exit()
        activation[sample_id] = squeezed_output
        
    return hook
def deep_swin_feat(load_path):
    ds = BratsDataset(root_dir, transform=transformer_transform)
    # ds=Subset(full_ds,np.sort(train_indices))
    image_paths = [('/').join(x[0].split('/')[:-1]) for x in ds.image_list]
    
    state_dict = torch.load(load_path)
    
    if 'state_dict' in state_dict:
        state_dict= state_dict['state_dict']
      
    model=SwinUNETREncoder(
        img_size=roi,
        feature_size=48,
        in_channels=4,
        out_channels=3,
        use_checkpoint=True
        )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    is_dataparallel = any(key.startswith('module.') for key in state_dict.keys())
    if is_dataparallel:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict,strict=True)
    model.eval()
    
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    activation = {}
    
    with torch.no_grad():
        for idx, test_data in enumerate(loader):
            test_inputs = test_data["image"].to(device)
            print("Input std value:", test_inputs.std().item())
            output= model(test_inputs)
            print("Output std value:", output.std().item())
            squeezed_output = torch.squeeze(torch.max(torch.max(torch.max(output, dim=4)[0], dim=3)[0], dim=2)[0]).detach().cpu().numpy()
            
            # print(squeezed_output.shape)
            activation[image_paths[idx]]= squeezed_output
    # Convert to DataFrame and save as CSV
    activation_df = pd.DataFrame.from_dict(activation, orient='index')
    activation_df.index.name = 'mask_path'
    day_month = datetime.now().date().strftime('%d%B')
    activation_df.to_csv(f'/scratch/a.bip5/BraTS/TfmrBotN_ft_{day_month}.csv')
    return None
    
def single_encode_tfmr_E10(test_inputs,model):
    cropper = CenterSpatialCrop(roi_size=(192, 192, 128))
    
    cropped_tensor = cropper(test_inputs[0])
    output= model(test_inputs)
    squeezed_output = torch.squeeze(torch.max(torch.max(torch.max(output, dim=4)[0], dim=3)[0], dim=2)[0]).detach().cpu().numpy()
    return squeezed_output
    
def swinUNETR_features(load_path):
    full_ds = BratsDataset(root_dir, transform=transformer_transform)
    # ds=Subset(full_ds,np.sort(train_indices))
    image_paths = [('/').join(x[0].split('/')[:-1]) for x in ds.image_list]
    # image_paths=[image_paths[x] for x in train_indices]
    # # #Load the model state dictionary
    # # state_dict = torch.load(load_path, map_location=torch.device('cpu'))
    # # print("State dictionary keys:")
    # # for key in state_dict.keys():
        # # print(key)
    
    model = model_loader(load_path)  # Initialize the model
    
    
    # print("Model named modules:")
    # for name, module in model.named_modules():
        # print(name)
    
    # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # if missing_keys:
        # print(f"Missing keys: {missing_keys}")
    # if unexpected_keys:
        # print(f"Unexpected keys: {unexpected_keys}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    model.eval()

    loader = DataLoader(full_ds, batch_size=1, shuffle=False, num_workers=4)
    activation = {}

    with torch.no_grad():
        for idx, test_data in enumerate(loader):
            test_inputs = test_data["image"].to(device)
            print("input max value:", test_inputs.max().item())
            # print(f"Test inputs shape: {test_inputs.shape}, device: {test_inputs.device}")
            hook= get_activation(image_paths[idx], activation)
            
            handle= model.encoder4.register_forward_hook(hook)
            # # Attach the hook with the sample ID and activation dictionary
            # layer_name = "conv2"
            # if hasattr(model.encoder10.layer, layer_name):
                # hook = get_activation(test_data['id'][0], activation)
                # handle = getattr(model.encoder10.layer, layer_name).register_forward_hook(hook)
                # # print(f"Hook registered to layer: {layer_name}")
            # else:
                # print(f"Layer {layer_name} does not exist in the model.")
                # continue
            
            # Forward pass
            _ = model(test_inputs)  # No need to print output here
            
            # Remove the hook after the forward pass
            handle.remove()
           

    # Save the activations
    np.save('activations.npy', activation)

    # Convert to DataFrame and save as CSV
    activation_df = pd.DataFrame.from_dict(activation, orient='index')
    activation_df.index.name = 'mask_path'
    day_month = datetime.now().date().strftime('%d%B')
    activation_df.to_csv(f'/scratch/a.bip5/BraTS/Tfmrft_{day_month}.csv')

    return None
    
def single_encode_tfmr(test_inputs,model):
    
    cropper = CenterSpatialCrop(roi_size=(192, 192, 128))
    
    cropped_tensor = cropper(test_inputs[0])
    activation={}
    hook = get_activation('placeholder', activation)
            
    handle = model.encoder4.register_forward_hook(hook)
    # Forward pass
    _ = model(cropped_tensor.unsqueeze(0)) 

    features = activation['placeholder'] # list(activation.values())[0] previously
    
    return features
     
    
        
def SegResNet_features(load_path ):
    activation={}
    ds = BratsDataset(root_dir, transform=test_transforms1)
    # ds=Subset(full_ds,np.sort(train_indices))
    image_paths=[('/').join(x[0].split('/')[:-1]) for x in ds.image_list]
    # image_paths=[image_paths[x] for x in train_indices]
    
    model= model_loader(load_path)
    device = torch.device("cuda:0")

    model.to(device)   

    model.eval()
    
    
      
    
    
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    
    
    
    all_features = []  # List to store features for all test images
    
    with torch.no_grad():
        for test_data in loader:  # Fixed variable name here
            test_inputs = test_data["image"].to(device)
            if hasattr(model,'module'):
                encoded_maps, _ = model.module.encode(test_inputs)
            else:
                encoded_maps, _ = model.encode(test_inputs)
            # features = torch.mean(encoded_maps, dim=(2,3,4),keepdim=True)
            # Reduce over the 4th dimension first
            max_val_4th_dim, _ = torch.max(encoded_maps, dim=4, keepdim=True)

            # Then reduce over the 3rd dimension
            max_val_3rd_dim, _ = torch.max(max_val_4th_dim, dim=3, keepdim=True)

            # Finally, reduce over the 2nd dimension
            features, _ = torch.max(max_val_3rd_dim, dim=2, keepdim=True)

            
            # Append features to the list
            all_features.append(features.cpu().numpy().flatten())
    
    feature_df= pd.DataFrame(data=np.array(all_features),index=image_paths[:])
    feature_df.index.name='mask_path'
    day_month=datetime.now().date().strftime('%d%B')
    feature_df.to_csv(f'/scratch/a.bip5/BraTS/ft_{day_month}_ts{temporal_split}.csv')
    
    return None
    
def single_encode(test_inputs,model):
    
   
    
    if hasattr(model,'module'):
        encoded_maps, _ = model.module.encode(test_inputs)
    else:
        encoded_maps, _ = model.encode(test_inputs)
    
    # features = torch.mean(encoded_maps, dim=(2,3,4),keepdim=True)
    # Reduce over the 4th dimension first
    max_val_4th_dim, _ = torch.max(encoded_maps, dim=4, keepdim=True)

    # Then reduce over the 3rd dimension
    max_val_3rd_dim, _ = torch.max(max_val_4th_dim, dim=3, keepdim=True)

    # Finally, reduce over the 2nd dimension
    features, _ = torch.max(max_val_3rd_dim, dim=2, keepdim=True)
    return features.cpu().numpy().flatten()


if __name__=='__main__':
    if model_name=='transformer':
        print('TRANSFORMERS')
        deep_swin_feat(load_path)
        
        # swinUNETR_features(load_path)
    else:
        SegResNet_features(load_path)
    