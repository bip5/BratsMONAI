from typing import List, Optional, Sequence, Tuple, Union
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform
from monai.utils import TransformBackends, convert_data_type, deprecated_arg, ensure_tuple, look_up_option
import pandas as pd
# print(pandas.__version__)
import nibabel as nb
import re
import numpy as np


from monai.losses import DiceLoss

import torch
import gc

import os
from monai.transforms import (
MapTransform
)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npy', '.gz'
]

# A source: Nvidia HDGAN
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# makes a list of all image paths inside a directory
def make_dataset(data_dir):
    all_files = []
    images=[]
    labels=[]
    im_temp=[]
    assert os.path.isdir(data_dir), '%s is not a valid directory' % data_dir
    
    for root, fol, _ in sorted(os.walk(data_dir)): # list folders and root
        for folder in fol:                    # for each folder
             path=os.path.join(root, folder)  # combine root path with folder path
             for root1, _, fnames in os.walk(path):       #list all file names in the folder         
                for f in fnames:                          # go through each file name
                    fpath=os.path.join(root1,f)
                    if is_image_file(f):                  # check if expected extension
                        if re.search("seg",f):            # look for the label files- have'seg' in the name 
                            labels.append(fpath)
                        else:
                            im_temp.append(fpath)         # all without seg are image files, store them in a list for each folder
                images.append(im_temp)                    # add image files for each folder to a list
                im_temp=[]
    return images, labels
    
    # A source: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 1 and label 4 to construct TC
            result.append(np.logical_or(d[key] == 1, d[key] == 4))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 4), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 4)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d
        
class Ensemble:
    @staticmethod
    def get_stacked_torch(img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> torch.Tensor:
        """Get either a sequence or single instance of np.ndarray/torch.Tensor. Return single torch.Tensor."""
        if isinstance(img, Sequence) and isinstance(img[0], np.ndarray):
            print("converting pred list to tensor")
            img = [torch.as_tensor(i) for i in img]
        elif isinstance(img, np.ndarray):
            print("converting pred list to tensor")  
            img = torch.as_tensor(img)
        # else:
            # print("conversion wasn't necessary")
        
        out: torch.Tensor = torch.stack(img) if isinstance(img, Sequence) else torch.unsqueeze(img ,dim=0) # type: ignore
        return out

    @staticmethod
    def post_convert(img: torch.Tensor, orig_img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> NdarrayOrTensor:
        orig_img_ = orig_img[0] if isinstance(orig_img, Sequence) else orig_img
        out, *_ = convert_to_dst_type(img, orig_img_)
        return out
        

class ConfEnsemble(Ensemble, Transform):


    backend = [TransformBackends.TORCH]

    def __init__(self, weights: Optional[Union[Sequence[float], NdarrayOrTensor]] = None) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.float) if weights is not None else None

    def __call__(self, img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> NdarrayOrTensor:
        img_ = self.get_stacked_torch(img)
        if self.weights is not None:
            self.weights = self.weights.to(img_.device)
            shape = tuple(self.weights.shape)
            for _ in range(img_.ndimension() - self.weights.ndimension()):
                shape += (1,)
            weights = self.weights.reshape(*shape)

            img_ = img_ * weights / weights.mean(dim=0, keepdim=True)
            
        
        img_=torch.sort(img_,dim=0,descending=True).values #img is a tensor        
        
        # print(img_[0,2,100,100,70],img_[1,2,100,100,70],img_[2,2,100,100,70],img_[3,2,100,100,70],img_[4,2,100,100,70])
        out_pt = torch.mean(img_[0:10,:,:,:,:], dim=0)#[0:3,:,:,:,:]
        # print(out_pt.shape)
        x=self.post_convert(out_pt, img)
        
       
        return x
        
class Ensembled(MapTransform):
   

    backend = list(set(ConfEnsemble.backend) )

    def __init__(
        self, keys,ensemble,
        output_key = None,
        allow_missing_keys = False,
    ) -> None:
   
        super().__init__(keys, allow_missing_keys)
        if not callable(ensemble):
            raise TypeError(f"ensemble must be callable but is {type(ensemble).__name__}.")
        self.ensemble = ensemble
        if len(self.keys) > 1 and output_key is None:
            raise ValueError("Incompatible values: len(self.keys) > 1 and output_key=None.")
        self.output_key = output_key if output_key is not None else self.keys[0]

    def __call__(self, data):
        d = dict(data)
        
        if len(self.keys) == 1 and self.keys[0] in d:
            items = d[self.keys[0]]
        else:
            items = [d[key] for key in self.key_iterator(d)]

        if len(items) > 0:
            d[self.output_key] = self.ensemble(items)

        return d
        
class ConfEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MeanEnsemble`.
    """

    backend = ConfEnsemble.backend

    def __init__(
        self,
        keys,
        output_key = None,
        weights = None,
    ) -> None:

        ensemble = ConfEnsemble(weights=weights)
        super().__init__(keys, ensemble, output_key)
                