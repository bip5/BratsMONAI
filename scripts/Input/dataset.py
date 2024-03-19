import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')

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
import monai

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
        for fname in sorted(fnames):
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

def make_atlas_dataset(data_dir):
    images = []
    masks = []
    im_temp = []

    assert os.path.isdir(data_dir), '%s is not a valid directory' % data_dir

    for root, _, fnames in sorted(os.walk(data_dir)):
        im_temp = []
        for fname in sorted(fnames):
            fpath = os.path.join(root, fname)
            if is_image_file(fname):
                if re.search("mask", fname):
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

    for folder in sorted(folders):                    # for each folder
         path=folder # combine root path with folder path
         for root1, _, fnames in sorted(os.walk(path)):       #list all file names in the folder         
            for f in sorted(fnames):                          # go through each file name
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

# def make_exp_dataset(path, sheet):
    # all_files = []
    # images = []
    # masks = []
    # folders = pd.read_excel(path, sheet)['Index']

    # for folder in sorted(folders):  # for each subject folder
        
        # if 'GLIValidationData' in folder:
            # continue
            
        # else:
        
            # im_temp = []  # reset the temporary list for image files for each subject
            # folder_path = os.path.join(path, folder)  # combine root path with folder path
            
            # for root1, _, fnames in sorted(os.walk(folder_path)):  # list all file names in the subject folder
                # for f in sorted(fnames):  # sort file names to ensure consistent modality order
                    # fpath = os.path.join(root1, f)
                    # if is_image_file(f):  # check if expected extension
                        # if re.search("seg", f):  # look for the mask files- have 'seg' in the name
                            # masks.append(fpath)
                        # else:
                            # im_temp.append(fpath)  # all without 'seg' are image files, store them in a list for each subject
                    
            # if im_temp:
                # images.append(im_temp)  # append the list of image files for the subject to the images list
    
    
    # return images, masks
    
def make_exp_dataset(path, sheet):
    all_files = []
    images = []
    masks = []
    folders = pd.read_excel(path, sheet)['Index'] # list of strings eg>'scratch/a.bip5/BraTS/BraTS_23_training/BraTS-GLI-00000-000'

    for folder in sorted(folders):  # for each subject folder
       
            im_temp = []  # reset the temporary list for image files for each subject
            # folder_path = os.path.join(path, folder)  # combine root path with folder path
            
            for root1, _, fnames in sorted(os.walk(folder)):  # list all file names in the subject folder
                for f in sorted(fnames):  # sort file names to ensure consistent modality order
                    fpath = os.path.join(root1, f)
                    if is_image_file(f):  # check if expected extension
                        if re.search("seg", f):  # look for the mask files- have 'seg' in the name
                            masks.append(fpath)
                        else:
                            im_temp.append(fpath)  # all without 'seg' are image files, store them in a list for each subject
                        
            if im_temp:
                images.append(im_temp)  # append the list of image files for the subject to the images list
    
    
    return images, masks

indexes=np.random.choice(np.arange(max_samples),max_samples,replace=False)
fold=int(max_samples/5)


####ONLY WORKS WHEN MAX SAMPLES==1250####
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
            
class AtlasDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        
        data=make_atlas_dataset(data_dir)
        
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
            
            if not isinstance(item_dict['image'], monai.data.meta_tensor.MetaTensor):
                raise TypeError("The transformed 'image' is not a MetaTensor. Please check your transforms.")

            if not isinstance(item_dict['mask'], monai.data.meta_tensor.MetaTensor):
                raise TypeError("The transformed 'mask' is not a MetaTensor. Please check your transforms.")
        
        return item_dict           
           
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
            
            if not isinstance(item_dict['image'], monai.data.meta_tensor.MetaTensor):
                raise TypeError("The transformed 'image' is not a MetaTensor. Please check your transforms.")

            # if not isinstance(item_dict['mask'], monai.data.meta_tensor.MetaTensor):
                # raise TypeError("The transformed 'mask' is not a MetaTensor. Please check your transforms.")
        
        return item_dict
def time_list(root_dir):
    images,masks=make_dataset(root_dir)
    old_index=[]
    new_index=[]

    old=[]
    new=[]
    for index, (iname, mname) in enumerate(zip(images,masks)):
        if '001-seg' in mname:
            new_index.append(index)
            new.append((iname,mname))            
            
            old_index.append(index-1)
            old.append((images[index-1], masks[index-1]))
    return old,new,old_index,new_index
class BratsTimeDataset(Dataset):
    def __init__(self,data_dir,tid,transform=None):
        
        data=time_list(data_dir)
        if tid==0:
            self.data_list=data[0]         
            self.data_index=data[2]
        else:
            self.data_list=data[1]         
            self.data_index=data[3]
            
        self.transform=transform
        
    def __len__(self):
#         return len(os.listdir(self.mask_dir))
        return min(max_samples,len(self.data_list))#
    
    def __getitem__(self,idx):
        # print(idx)
       
        image=self.data_list[idx][0]
       
    
        mask=self.data_list[idx][1] 
        

            
        item_dict={"image":image,"mask":mask}
        # print(item_dict)
        
        if self.transform:
            item_dict={"image":image,"mask": mask}
            
            item_dict=self.transform(item_dict)
            item_dict['id'] = mask[-30:-11]
            item_dict['index'] = self.data_index[idx]
            
            if not isinstance(item_dict['image'], monai.data.meta_tensor.MetaTensor):
                raise TypeError("The transformed 'image' is not a MetaTensor. Please check your transforms.")

            # if not isinstance(item_dict['mask'], monai.data.meta_tensor.MetaTensor):
                # raise TypeError("The transformed 'mask' is not a MetaTensor. Please check your transforms.")
        
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
        
        # print(len(self.image_list),'making sure getting all 300') #this should work
        self.transform=transform
        
    def __len__(self):
#         return len(os.listdir(self.mask_dir))
        
        return min(max_samples,len(self.image_list))#
    
    def __getitem__(self,idx):
        # print(idx)
        
        image=self.image_list[idx]
    
        mask=self.mask_list[idx] 
        
            
        item_dict={"image":image,"mask":mask}
        
        if self.transform:            
            item_dict2=self.transform(item_dict)
            item_dict2['id'] = image[0][-20:-11]
            item_dict2['imagepaths']=image
            # print(image)
            # print(type(item_dict2['image']))
            if not isinstance(item_dict2['image'], monai.data.meta_tensor.MetaTensor):
                raise TypeError("The transformed 'image' is not a MetaTensor. Please check your transforms.")

            if not isinstance(item_dict2['mask'], monai.data.meta_tensor.MetaTensor):
                raise TypeError("The transformed 'mask' is not a MetaTensor. Please check your transforms.")
            
        
        return item_dict2

class ExpDatasetEval(Dataset):
    def __init__(self,path,sheet,transform=None):
        
        self.image_list,_=make_exp_dataset(path,sheet)
        
        
        self.transform=transform
        
    def __len__(self):
#         return len(os.listdir(self.mask_dir))
        return min(max_samples,len(self.image_list))#
    
    def __getitem__(self,idx):
        # print(idx, 'dataset index')
        # print(len(self.image_list),'len(self.image_list)')
        image=self.image_list[idx]
    
        
        
            
        item_dict={"image":image}
        
        if self.transform:            
            item_dict2=self.transform(item_dict)
            item_dict2['id'] = image[0][-20:-11]
            item_dict2['imagepaths']=image
            # print(image)
            # print(type(item_dict2['image']))
            if not isinstance(item_dict2['image'], monai.data.meta_tensor.MetaTensor):
                raise TypeError("The transformed 'image' is not a MetaTensor. Please check your transforms.")

           
            
        
        return item_dict2

        
class Brats23valDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        
        data=make_dataset(data_dir)[0]
        extension=make_dataset('/scratch/a.bip5/BraTS/GLIValidationData')[0]
        data.extend(extension) # extending data list)
        self.image_list=data
        
        self.transform=transform
        
    def __len__(self):
#         return len(os.listdir(self.mask_dir))
        return len(self.image_list)#
    
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
    