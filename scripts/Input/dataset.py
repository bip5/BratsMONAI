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
    all_files = []
    images=[]
    masks=[]
    im_temp=[]
    assert os.path.isdir(data_dir), '%s is not a valid directory' % data_dir
    
    for root, fol, _ in sorted(os.walk(data_dir)): # list folders and root
        for folder in fol:                    # for each folder
             path=os.path.join(root, folder)  # combine root path with folder path
             for root1, _, fnames in os.walk(path):       #list all file names in the folder         
                for f in fnames:                          # go through each file name
                    fpath=os.path.join(root1,f)
                    if is_image_file(f):                  # check if expected extension
                        if re.search("seg",f):            # look for the mask files- have'seg' in the name 
                            masks.append(fpath)
                        else:
                            im_temp.append(fpath)         # all without seg are image files, store them in a list for each folder
                images.append(im_temp)                    # add image files for each folder to a list
                im_temp=[]
    return images, masks
    

indexes=np.random.choice(np.arange(max_samples),max_samples,replace=False)
fold=int(max_samples/5)

for i in range(1,6):
    if i==int(fold_num):
        if i<5:
            val_indices=indexes[(i-1)*fold:i*fold]
            train_indices=np.delete(indexes,val_indices)#indexes[i*fold:(i+1)*fold]#
        else:
            val_indices=indexes[(i-1)*fold:i*fold]
            train_indices=np.delete(indexes,val_indices)#indexes[(i-5)*fold:(i-4)*fold]
            
           
           
class BratsDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        
        self.image_list=make_dataset(data_dir)[0]
         
        self.mask_list=make_dataset(data_dir)[1] 
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
            
        
        return item_dict



    
    