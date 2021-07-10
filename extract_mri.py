#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# challenege BraTS 2021
# this code is to convert nibabels format MRI data to other formats to visualise or use in dep learning based models.

# Written by Azam Hamidinekoo July 2021
# -----------------------------------------------------------------------------
import numpy as np
import os

# file manipulation
from skimage.io import imsave
import scipy.io as sio
import matplotlib.pyplot as plt
# niftii support
import nibabel as nib


# variant:   1 - png,  2 - numpy,   3 - mat
variant = 1
dir_nameoutput = 'dataset_exported_png' #change it into png or npy based on the type you want to extract the data
imshow_data = 0  # show th eplots ere to: 0-no  1-yes
dir_name = 'dataset'
dir_sep = '/'
dir_top = '.' + dir_sep + dir_name
data_types = ['_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz', '_flair.nii.gz', '_seg.nii.gz']
data_types_str = ['T1', 'T1ce', 'T2', 'FLAIR','GT']

for x in os.walk(dir_top):
    if x[0] != dir_name:
        print(x[0] + '\n')

    filename = x[0].split(dir_sep)
    if len(filename) > 2:
        # make subject directory
        #print('.' + dir_sep + dir_nameoutput + dir_sep + filename[2])
        os.mkdir('.' + dir_sep + dir_nameoutput + dir_sep + filename[2])
        # iterate over modalities
        for mm in range(0, len(data_types_str)):  
            # load data
            filename_t1 = x[0] + dir_sep + filename[2] + data_types[mm]
            img = nib.load(filename_t1)
            data_volume = img.get_fdata() 
            # iterate over all slices
            for ss in range(0, data_volume.shape[2]):
                #image = np.rot90(data_volume[:,:,ss], -1)
                image = data_volume[:,:,ss]
                if imshow_data == 1:
                    plt.imshow(image,cmap='gray')
                    plt.show()
                # used for numpy & mat files
                dict_data = {'image': image}
                if not os.path.isdir('.' + dir_sep + dir_nameoutput + dir_sep + filename[2] + dir_sep + data_types_str[mm]):
                    os.mkdir('.' + dir_sep + dir_nameoutput + dir_sep + filename[2] + dir_sep + data_types_str[mm])
                filename_name = '.' + dir_sep + dir_nameoutput + dir_sep + filename[2] + dir_sep + data_types_str[mm] + dir_sep + str(ss+1)
                #----------------------------------------------------------------------------------------
                # PNG
                if variant == 1:
                    imsave(filename_name + '.png', image)
                # NUMPY                    
                elif variant == 2:
                    np.save(filename_name + '.npy', dict_data)
                # MAT                    
                else:
                    sio.savemat(filename_name + '.mat', dict_data)  
            else:
                print(" This image is mostly black. This image is discarded")
        
