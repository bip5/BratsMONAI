from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
from monai.engines import SupervisedTrainer
from torch.nn.parallel import DistributedDataParallel
import pandas
import nibabel
import numpy as np
from monai.transforms import (CastToTyped,
                              CropForegroundd, EnsureChannelFirstd, 
                              NormalizeIntensity, RandCropByPosNegLabeld,                              
                              RandGaussianSmoothd, 
                              RandZoomd, SpatialCrop, SpatialPadd)
from monai.transforms.compose import MapTransform
from monai.transforms.utils import generate_spatial_bounding_box
from skimage.transform import resize
import os
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from monai.config import print_config
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from monai.data import (CacheDataset, DataLoader, load_decathlon_datalist,
                        load_decathlon_properties, partition_dataset)
from monai.networks.nets import DynUNet
import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine
from ignite.metrics import Metric
from monai.data import decollate_batch
from monai.data.nifti_writer import write_nifti
from monai.engines import SupervisedEvaluator
from monai.engines.utils import IterationEvents, default_prepare_batch
from monai.inferers import Inferer
from monai.networks.utils import eval_mode
from monai.transforms import AsDiscrete, Transform
from torch.utils.data import DataLoader

# from task_params import clip_values, normalize_values, patch_size, spacing

import monai
from monai.data import Dataset
from monai.utils import set_determinism
from monai.apps import CrossValidation
from monai.engines.utils import CommonKeys as Keys
from monai.engines.utils import IterationEvents, default_prepare_batch
from monai.inferers import Inferer
from monai.networks.utils import eval_mode
from monai.transforms import AsDiscrete, Transform
from torch.utils.data import DataLoader



from monai.transforms import (EnsureChannelFirstD, AddChannelD,\
    ScaleIntensityD, SpacingD, OrientationD,\
    ResizeD, RandAffineD,
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    RandBiasFieldD,
    RandRotateD,
    RotateD, Rotate,
    RandGaussianSmoothD,
    RandGaussianNoised,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd, RandFlip,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,   
    EnsureTyped,
    EnsureType,
)



from monai.losses import DiceLoss
from monai.utils import UpsampleMode
from monai.data import decollate_batch, list_data_collate

from monai.networks.nets import SegResNet, UNet, DynUNet
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import DataLoader
import numpy as np
from datetime import date, datetime
import sys
import re
import time
import argparse
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import Subset




        
if __name__=="__main__":

    parser=argparse.ArgumentParser(description="Monai Seg main")

    parser.add_argument("--lr",default=1e-3,type=float,help="learning rate")
    parser.add_argument("--model",default="SISANet",type=str,help="name of model to use")
    parser.add_argument("--load_save",default =1, type=int,help="flag to use saved model weight")
    parser.add_argument("--load_path",default="/scratch/a.bip5/BraTS 2021/2022-01-20T16best_metric_model.pth", type=str, help="file path to load previously saved model")
    parser.add_argument("--batch_size",default=2, type=int, help="to define batch size")
    parser.add_argument("--save_name", default="SISANET.pth",type=str, help="save name")
    parser.add_argument("--upsample", default="DECONV",type=str, help="flag to choose deconv options- NONTRAINABLE, DECONV, PIXELSHUFFLE")
    parser.add_argument("--barlow_final",default=1, type=int, help="flag to use checkpoint instead of final model for barlow")
    parser.add_argument("--bar_model_name",default="checkpoint.pth", type=str,help="model name to load")
    parser.add_argument("--max_samples",default=10000,type=int,help="max number of samples to use for training")
    parser.add_argument("--fold_num",default=1,type=str,help="cross-validation fold number")
    parser.add_argument("--epochs",default=150,type=int,help="number of epochs to run")
    parser.add_argument("--CV_flag",default=0,type=int,help="is this a cross validation fold? 1=yes")
    parser.add_argument("--seed",default=0,type=int, help="random seed for the script")
    parser.add_argument("--method",default='A', type=str,help='A,B or C')
    parser.add_argument("--workers",default=8, type=int,help='number of cpu threads')
    args=parser.parse_args()

    print(' '.join(sys.argv))
    
    os.environ['PYTHONHASHSEED']=str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    set_determinism(seed=args.seed)

    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npy', '.gz'
    ]


    set_determinism(seed=0)

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
        


    # A source: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
    class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
        """
        Convert masks to multi channels based on brats classes:
        mask 1 is the peritumoral edema
        mask 2 is the GD-enhancing tumor
        mask 3 is the necrotic and non-enhancing tumor core
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor).

        """

        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                result = []
                # merge mask 2 and mask 3 to construct TC
                result.append(np.logical_or(d[key] == 2, d[key] == 3))
                # merge masks 1, 2 and 3 to construct WT
                result.append(
                    np.logical_or(
                        np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                    )
                )
                # mask 2 is ET
                result.append(d[key] == 2)
                d[key] = np.stack(result, axis=0).astype(np.float32)
            return d


    indexes=np.arange(args.max_samples)
    fold=int(args.max_samples/5)

    for i in range(1,6):
        if i==int(args.fold_num):
            if i<5:
                val_indices=indexes[(i-1)*fold:i*fold]
                train_indices=np.delete(indexes,val_indices)#indexes[i*fold:(i+1)*fold]#
            else:
                val_indices=indexes[(i-1)*fold:i*fold]
                train_indices=indexes[(i-5)*fold:(i-4)*fold]#np.delete(indexes,val_indices)#
                
               
    class BratsDataset(Dataset):
        def __init__(self,data_dir,transform=None):
            
            self.image_list=make_dataset(data_dir)[0]
             
            self.mask_list=make_dataset(data_dir)[1] 
            self.transform=transform
            
        def __len__(self):
    #         return len(os.listdir(self.mask_dir))
            return min(args.max_samples,len(self.mask_list))#
        
        def __getitem__(self,idx):
            # print(idx)
           
            image=self.image_list[idx]
        
            mask=self.mask_list[idx] 

                
            item_dict={"image":image,"mask":mask}
            
            if self.transform:
                item_dict={"image":image,"mask": mask}
                item_dict=self.transform(item_dict)
                
            
            return item_dict




        
    KEYS=("image","mask")
    print("Transforms not defined yet")
    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstD(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
            SpacingD(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            OrientationD(keys=["image", "mask"], axcodes="RAS"),
            RandSpatialCropd(keys=["image", "mask"], roi_size=[192, 192, 144], random_size=False),
           
            RandRotateD(keys=["image","mask"],range_x=0.1,range_y=0.1, range_z=0.1,prob=0.5),
           
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            EnsureTyped(keys=["image", "mask"]),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstD(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
            SpacingD(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            OrientationD(keys=["image", "mask"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandSpatialCropd(keys=["image", "mask"], roi_size=[192, 192, 144], random_size=False),
            EnsureTyped(keys=["image", "mask"]),
        ]
    )

    #dataset=DecathlonDataset(root_dir="/scratch/a.bip5/BraTS 2021/", task="Task05_Prostate",section="training", transform=xform, download=True)
    train_dataset=BratsDataset("/scratch/a.bip5/BraTS 2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData"  ,transform=train_transform ) 

    


    if args.CV_flag==1:
        print("loading cross val data")
        val_dataset=Subset(train_dataset,val_indices)
        train_dataset=Subset(train_dataset,train_indices)
        
    # else:     
        # print("loading data for single model training")
        # val_dataset=Subset(train_dataset,np.arange(800,1000))
        # train_dataset=Subset(train_dataset,np.arange(800))
        
        
    print("number of files processed: ", train_dataset.__len__())
    train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.workers)
    val_loader=DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)
    print("All Datasets assigned")

    root_dir="/scratch/a.bip5/BraTS 2021/"

    max_epochs = args.epochs
    val_interval = 1
    VAL_AMP = True
    
    def get_kernels_strides():
    
  
        sizes, spacings = [128, 128, 128], [1.0, 1.0, 1.0]
        input_size = sizes
        strides, kernels = [], []
        while True:
            spacing_ratio = [sp / min(spacings) for sp in spacings]
            stride = [
                2 if ratio <= 2 and size >= 8 else 1
                for (ratio, size) in zip(spacing_ratio, sizes)
            ]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            for idx, (i, j) in enumerate(zip(sizes, stride)):
                if i % j != 0:
                    raise ValueError(
                        f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                    )
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)

        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])
        print(kernels,strides,'kernels,strides')
        return kernels, strides
    
    kernels,strides=get_kernels_strides()
    # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    
    torch.manual_seed(args.seed)
    if args.model=="UNet":
         model=UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(64,128,256,512,1024),
            strides=(2,2,2,2)
            ).to(device)
    elif args.model=="SegResNet":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=32,
            norm="instance",
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[args.upsample]    
            ).to(device)
            
        model2 = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            norm="instance",
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[args.upsample]    
            ).to(device)
            
    elif args.model=="DynUNet":
        model=DynUNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=True,
            deep_supr_num=3,
        )

    else:
        model = locals() [args.model](4,3).to(device)

    with torch.cuda.amp.autocast():
        summary(model,(4,192,192,144))

    model=torch.nn.DataParallel(model)
    print("Model defined and passed to GPU")
    
    if args.load_save==1:
        model.load_state_dict(torch.load("/scratch/a.bip5/BraTS 2021/saved models/"+args.load_path),strict=False)
        print("loaded saved model ", args.load_path)

    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    post_trans = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        
    def inference(input):

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(192,192, 144),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)


    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    total_start = time.time()

    print("starting epochs")
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        
        torch.manual_seed(args.seed+1)
        model2.train()
        
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, masks = (
                batch_data["image"].to(device),
                batch_data["mask"].to(device),
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                outputs2=model2(inputs)
                simloss=loss_function(outputs,outputs2)
                loss = 2*loss_function(outputs, masks)*loss_function(outputs2, masks)* simloss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            if step%10==0:
                print(
                    f"{step}/{len(train_dataset) // train_loader.batch_size}"
                    f", train_loss: {loss.item():.4f}"
                    f", step time: {(time.time() - step_start):.4f}"
                )
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        # if epoch>50:
            # torch.save(
                            # model.state_dict(),
                            # os.path.join(root_dir, date.today().isoformat()+'T'+str(datetime.today().hour)+ args.model+"ep"+str(epoch+1)))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():

                for val_data in val_loader:
                    val_inputs, val_masks = (
                        val_data["image"].to(device),
                        val_data["mask"].to(device),
                    )
                    val_outputs = inference(val_inputs)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_masks)
                    dice_metric_batch(y_pred=val_outputs, y=val_masks)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                metric_tc = metric_batch[0].item()
                metric_values_tc.append(metric_tc)
                metric_wt = metric_batch[1].item()
                metric_values_wt.append(metric_wt)
                metric_et = metric_batch[2].item()
                metric_values_et.append(metric_et)
                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    if args.CV_flag==1:
                        torch.save(
                            model.state_dict(),
                            os.path.join(root_dir, args.model+"CV"+str(args.fold_num)+"ms"+str(args.max_samples)+"rs"+str(args.seed)+args.method)
                        )
                    else:
                        torch.save(
                            model.state_dict(),
                            os.path.join(root_dir, date.today().isoformat()+'T'+str(datetime.today().hour)+ args.model),
                        )
                    print("saved new best metric model")
                    
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                

                
        print(f"time consumption of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
    with open ('./time_consumption.csv', 'a') as sample:
        sample.write(f"{args.model},{args.method},{total_time},{date.today().isoformat()},{args.fold_num},{args.CV_flag},{args.seed}\n")
        print("File write sucessful")
