import sys
sys.path.append('/scratch/a.bip5/BraTS 2021/scripts/')

import os
import re
import numpy as np
from Input.config import ( 
fold_num,
max_samples,
seed,
)
from torch.utils.data import Subset
from monai.data import Dataset
import pandas as pd


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npy', '.gz'
]


np.random.seed(seed)


# A source: Nvidia HDGAN
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# makes a list of all image paths inside a directory
def make_dataset(data_dir):
    images = []
    masks = []
    im_temp = []

    assert os.path.isdir(data_dir), '%s is not a valid directory' % data_dir

    for root, _, fnames in sorted(os.walk(data_dir)):
        im_temp = []
        for fname in fnames:
            fpath = os.path.join(root, fname)
            if is_image_file(fname):
                if re.search("seg", fname):
                    masks.append(fpath)
                    # print(fpath)  # For debugging
                else:
                    im_temp.append(fpath)
        if im_temp:
            images.append(im_temp)

    return images, masks


def make_ens_dataset(path):
    all_files = []
    images=[]
    masks=[]
    im_temp=[]
    folders=pd.read_csv(path)['mask_path']

    for folder in folders:                    # for each folder
         path=folder # combine root path with folder path
         for root1, _, fnames in os.walk(path):       #list all file names in the folder         
            for f in fnames:                          # go through each file name
                fpath=os.path.join(root1,f)
                if is_image_file(f):                  # check if expected extension
                    if re.search("seg",f):            # look for the mask files- have'seg' in the name 
                        masks.append(fpath)
                    else:
                        im_temp.append(fpath)         # all without seg are image files, store them in a list for each folder
            if im_temp:            
                    images.append(im_temp)                    # add image files for each folder to a list
                    im_temp=[]
    return images, masks

def make_exp_dataset(path,sheet):
    all_files = []
    images=[]
    masks=[]
    im_temp=[]
    folders=pd.read_excel(path,sheet)['Index']

    for folder in folders:                    # for each folder
         path=folder # combine root path with folder path
         for root1, _, fnames in os.walk(path):       #list all file names in the folder         
            for f in fnames:                          # go through each file name
                fpath=os.path.join(root1,f)
                if is_image_file(f):                  # check if expected extension
                    if re.search("seg",f):            # look for the mask files- have'seg' in the name 
                        masks.append(fpath)
                    else:
                        im_temp.append(fpath)         # all without seg are image files, store them in a list for each folder
            if im_temp:            
                    images.append(im_temp)                    # add image files for each folder to a list
                    im_temp=[]
    return images, masks

indexes=np.random.choice(np.arange(max_samples),max_samples,replace=False)
fold=int(max_samples/5)



for i in range(1,6):
    if i==int(fold_num):
        val_start=(i-1)*fold
        val_end=val_start+100
        test_start=val_end
        test_end=i*fold       
        
        val_indices=indexes[val_start:val_end] 
        test_indices=indexes[test_start:test_end]
        # print(test_indices)
        train_indices=np.delete(indexes,np.arange(val_start,test_end))
            
           
           
class BratsDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        
        data=make_dataset(data_dir)
        
        self.image_list=data[0]
         
        self.mask_list=data[1]
        self.transform=transform
        
    def __len__(self):
#         return len(os.listdir(self.mask_dir))
        return min(max_samples,len(self.mask_list))#
    
    def __getitem__(self,idx):
        # print(idx)
       
        image=self.image_list[idx]
       
    
        mask=self.mask_list[idx] 
        

            
        item_dict={"image":image,"mask":mask}
        # print(item_dict)
        
        if self.transform:
            item_dict={"image":image,"mask": mask}
            item_dict=self.transform(item_dict)
            item_dict['id'] = mask[-30:-11]
        
        return item_dict


class EnsembleDataset(Dataset):
    def __init__(self,csv_path,transform=None):
        
        data=make_ens_dataset(csv_path)
        self.image_list=data[0]
         
        self.mask_list=data[1] 
        # print('files processed:' , self.mask_list)
        self.transform=transform
        
    def __len__(self):
#         return len(os.listdir(self.mask_dir))
        return min(max_samples,len(self.mask_list))#
    
    def __getitem__(self,idx):
        # print(idx)
       
        image=self.image_list[idx]
    
        mask=self.mask_list[idx] 

            
        item_dict={"image":image,"mask":mask}
        
        if self.transform:
            item_dict={"image":image,"mask": mask}
            item_dict=self.transform(item_dict)
            item_dict['id'] = mask[-16:-11]
        
        return item_dict

class ExpDataset(Dataset):
    def __init__(self,path,sheet,transform=None):
        
        self.image_list,self.mask_list=make_exp_dataset(path,sheet)
        
        # print('files processed:' , self.mask_list)
        self.transform=transform
        
    def __len__(self):
#         return len(os.listdir(self.mask_dir))
        return min(max_samples,len(self.mask_list))#
    
    def __getitem__(self,idx):
        # print(idx)
       
        image=self.image_list[idx]
    
        mask=self.mask_list[idx] 

            
        item_dict={"image":image,"mask":mask}
        
        if self.transform:
            item_dict={"image":image,"mask": mask}
            item_dict2=self.transform(item_dict)
            item_dict2['id'] = mask[-16:-11]
            
        
        return item_dict2


class ExpDatasetEval(Dataset):
    def __init__(self,path,sheet,transform=None):
        
        self.image_list,self.mask_list=make_exp_dataset(path,sheet)
        
        # print('files processed:' , self.mask_list)
        self.transform=transform
        
    def __len__(self):
#         return len(os.listdir(self.mask_dir))
        return min(max_samples,len(self.mask_list))#
    
    def __getitem__(self,idx):
        # print(idx)
       
        image=self.image_list[idx]
    
        mask=self.mask_list[idx] 

            
        item_dict={"image":image,"label":mask}
        # print(item_dict)
        
        if self.transform:
            item_dict={"image":image,"label": mask}
            item_dict=self.transform(item_dict)
            
        
        return item_dict
        
class Brats23valDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        
        data=make_dataset(data_dir)
        
        self.image_list=data[0]
        
        self.transform=transform
        
    def __len__(self):
#         return len(os.listdir(self.mask_dir))
        return min(max_samples,len(self.image_list))#
    
    def __getitem__(self,idx):
        # print(idx)
       
        image=self.image_list[idx]
       
            
        item_dict={"image":image}
        # print(item_dict)
        
        if self.transform:
            item_dict={"image":image}
            item_dict=self.transform(item_dict)
            item_dict['id'] = image[0][-20:-11]
        
        return item_dict
    