import torch
import numpy as np
import os
from Input.dataset import BratsDataset
from Input.config import (
root_dir,
)
from Input.localtransforms import test_transforms1
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
import gzip
import multiprocessing
from functools import partial



#Dataset: this will be the standard BraTSDataset except we want to do nothing other than normalise(test transforms should do)
dataset = BratsDataset(root_dir,transform=test_transforms1)

#Dataloader: we want to create a dataloader that feeds one sample at a time to the clustering function.
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#Clustering function: cluster


def create_spatial_encoding(shape):
    '''Generate index for each pixel in a 3D image from a training pipeline'''
   
    z, y, x = [np.linspace(-1, 1, num=s) for s in shape[2:]]
    return np.stack(np.meshgrid(z, y, x, indexing='ij'), axis=0)
    
def concatenate_image_and_spatial_encoding(image):
    '''Generate spatial encoding for positions in a 3D image and concatenate'''
    spatial_encoding = create_spatial_encoding(image.shape) # assuming shape of BCHWD
   
    spatial_enc_reshaped = spatial_encoding.reshape(-1,spatial_encoding.shape[0])
    # Reshape image to (num_voxels, channels)
    image_reshaped = image.view(-1, image.shape[1]).cpu().numpy()  # assuming shape of BCHWD
    
    image_reshaped = np.concatenate((image_reshaped,spatial_enc_reshaped),axis=1)  
    return image_reshaped

def extract_clustered_image(clustered_image):
    '''Separate clustered image from the spatial coordinates used in clustering'''
    extracted_image = clustered_image[:,:4]
    return extracted_image

def save_map(output_dir,image_id,clustered_image_tensor ):
    '''Save to desired output dir'''
    os.makedirs(output_dir,exist_ok=True)
    save_path = os.path.join(output_dir, f"{image_id}_clustered.pt.gz")
    with gzip.open(save_path, 'wb') as f:
        torch.save(clustered_image_tensor, f)
    
    print(f"Saved clustered image to {save_path}")
    

def cluster_and_save(item_dict, n_clusters=4, output_dir='/scratch/a.bip5/BraTS/cluster_maps'):
    '''Cluster each 3D image and save a membership map'''
    image = item_dict['image']
    image_id = item_dict['id'][0]   
    save_path = os.path.join(output_dir, f"{image_id}_clustered.pt.gz")
    if os.path.exists(save_path):
        return None
    image_reshaped = concatenate_image_and_spatial_encoding(image)     
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(image_reshaped)
    
    memberships = kmeans.predict(image_reshaped)
    
    # Replace membership values with corresponding cluster centers
    clustered_image = kmeans.cluster_centers_[memberships]
    
    extracted_image = extract_clustered_image(clustered_image).reshape(image.shape)
    
    # Convert the result back to a PyTorch tensor
    clustered_image_tensor = torch.tensor(extracted_image)
    print(clustered_image_tensor.shape) #making sure tensors are the right shape
    
    # Save the clustered image
    save_map(output_dir,image_id,clustered_image_tensor )

def process_item(item_dict,i):
    print(f'Processing image {i}')
    cluster_and_save(item_dict)

def parallel_processing(dataloader, num_workers=None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()  # Use all available CPU cores
    
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(process_item, enumerate(dataloader))

if __name__ == '__main__':
    # Assuming your dataloader is already defined
    # parallel_processing(dataloader) ## NEEDS fixing


    for i,item_dict in enumerate(dataloader):
        print('processing image ',i)
        cluster_and_save(item_dict)