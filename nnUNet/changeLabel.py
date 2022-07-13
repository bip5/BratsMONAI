from copy import deepcopy
from multiprocessing.pool import Pool

import numpy as np
from collections import OrderedDict

from batchgenerators.utilities.file_and_folder_operations import *
#from meddec.paper_plot.nature_methods.challenge_visualization_stuff.own_implementation.ranking import \
#    rank_then_aggregate
import scipy.stats as ss

from nnunet.dataset_conversion.Task032_BraTS_2018 import convert_labels_back_to_BraTS_2018_2019_convention
from nnunet.evaluation.region_based_evaluation import get_brats_regions, evaluate_regions
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk
import shutil
from medpy.metric import dc, hd95

from nnunet.postprocessing.consolidate_postprocessing import collect_cv_niftis
from typing import Tuple
import os


def copy_BraTS_segmentation_and_convert_labels(in_file, out_file):
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3

    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)
    
for file in os.listdir("./Task500_BraTS2021/labelsTs"):
    out_file="./Task500_BraTS2021/labelsTs/"+file
    copy_BraTS_segmentation_and_convert_labels(out_file, out_file)