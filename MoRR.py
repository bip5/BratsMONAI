import pandas
# print(pandas.__version__)
import nibabel

import os
import monai
from monai.data import Dataset
from monai.utils import set_determinism
from monai.apps import CrossValidation

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

from monai.networks.nets import SegResNet, UNet
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import DataLoader
import numpy as np
from datetime import date, datetime
import sys
import re
import torch
import torch.nn as nn
import time
import argparse
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import Subset


os.environ['PYTHONHASHSEED']=str(0)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
set_determinism(seed=0)

        
if __name__=="__main__":

    parser=argparse.ArgumentParser(description="Monai Seg main")

    parser.add_argument("--lr",default=1e-3,type=float,help="learning rate")
    parser.add_argument("--model",default="SISANet",type=str,help="name of model to use")
    parser.add_argument("--load_save",default =1, type=int,help="flag to use saved model weight")
    parser.add_argument("--load_path",default="./2022-01-20T16best_metric_model.pth", type=str, help="file path to load previously saved model")
    parser.add_argument("--batch_size",default=2, type=int, help="to define batch size")
    parser.add_argument("--save_name", default="SISANET.pth",type=str, help="save name")
    parser.add_argument("--upsample", default="DECONV",type=str, help="flag to choose deconv options- NONTRAINABLE, DECONV, PIXELSHUFFLE")
    parser.add_argument("--barlow_final",default=1, type=int, help="flag to use checkpoint instead of final model for barlow")
    parser.add_argument("--bar_model_name",default="checkpoint.pth", type=str,help="model name to load")
    parser.add_argument("--max_samples",default=10000,type=int,help="max number of samples to use for training")
    parser.add_argument("--fold_num",default=1,type=str,help="cross-validation fold number")
    parser.add_argument("--epochs",default=150,type=int,help="number of epochs to run")
    parser.add_argument("--CV_flag",default=0,type=int,help="is this a cross validation fold? 1=yes")
    parser.add_argument("--bunch",default=10, type=int, help="how many data samples to bunch together to feed to each model")

    args=parser.parse_args()

    print(' '.join(sys.argv))

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
                train_indices=indexes[(i-1)*fold:i*fold]
                val_indices=indexes[i*fold:(i+1)*fold]#np.delete(indexes,train_indices)
            else:
                train_indices=indexes[(i-1)*fold:i*fold]
                val_indices=indexes[(i-5)*fold:(i-4)*fold]
                
               
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
    
    
    root_dir="./"

    max_epochs = args.epochs
    val_interval = 1
    VAL_AMP = True

    # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")

    if args.model=="UNet":
         model=UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(64,128,256,512,1024),
            strides=(2,2,2,2)
            ).to(device)
    elif args.model=="SegResNet":
        model1 = SegResNet(
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
            init_filters=32,
            norm="instance",
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[args.upsample]    
            ).to(device)
        model3 = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=32,
            norm="instance",
            in_channels=4,
            out_channels=3,
            upsample_mode=UpsampleMode[args.upsample]    
            ).to(device)
            

    else:
        model = locals() [args.model](4,3).to(device)

    with torch.cuda.amp.autocast():
        summary(model1,(4,192,192,144))

    model1=torch.nn.DataParallel(model1)
    model2=torch.nn.DataParallel(model2)
    model3=torch.nn.DataParallel(model3)
    print("Model defined and passed to GPU")

    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer1 = torch.optim.Adam(model1.parameters(), args.lr, weight_decay=1e-5)
    optimizer2 = torch.optim.Adam(model2.parameters(), args.lr, weight_decay=1e-5)
    optimizer3 = torch.optim.Adam(model3.parameters(), args.lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=max_epochs)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    post_trans = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        
    def inference1(input):

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(192,192, 144),
                sw_batch_size=1,
                predictor=model1,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)
            
    def inference2(input):

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(192,192, 144),
                sw_batch_size=1,
                predictor=model2,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)
            
    def inference3(input):

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(192,192, 144),
                sw_batch_size=1,
                predictor=model3,
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

    #dataset=DecathlonDataset(root_dir="./", task="Task05_Prostate",section="training", transform=xform, download=True)
    train_dataset=BratsDataset("./RSNA_ASNR_MICCAI_BraTS2021_TrainingData"  ,transform=train_transform ) 
 
    indexes=np.arange(train_dataset.__len__())
    

        
        
        
        

        

        
        



    
    


    
    best_metric11 = -1 
    best_metric12 = -1 
    best_metric13 = -1
    
    best_metric_epoch1 = -1
    
    best_metric21 = -1 
    best_metric22 = -1 
    best_metric23 = -1
    
    best_metric_epoch2 = -1
    
    best_metric = -1 
    best_metric32 = -1 
    best_metric33 = -1
    
    best_metric_epoch = 0
    
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values1 = []
    metric_values_tc1 = []
    metric_values_wt1 = []
    metric_values_et1 = []
    
    metric_values2 = []
    metric_values_tc2 = []
    metric_values_wt2 = []
    metric_values_et2 = []
    
    metric_values3 = []
    metric_values_tc3 = []
    metric_values_wt3 = []
    metric_values_et3 = []    
    

    total_start = time.time()

    print("starting epochs")
    bunch=args.bunch
    all_indices=np.arange(len(train_dataset))
    for epoch in range(max_epochs): # For a given number of epochs
        np.random.shuffle(all_indices)
        max_index=0
        indices0=all_indices[0:bunch]
        indices1=all_indices[bunch:2*bunch]
        indices2=all_indices[2*bunch:3*bunch] #get the data bunchs to feed to each model
        indices3=all_indices[3*bunch:4*bunch] 
        max_index=4*bunch
        epoch_start = time.time()
        
        
        while max_index<train_dataset.__len__():
           
              # get the indices to pass to each model
            
            
            look_start = time.time()
            train_dataset0=Subset(train_dataset,indices0)
            train_dataset1=Subset(train_dataset,indices1)
            train_dataset2=Subset(train_dataset,indices2)
            train_dataset3=Subset(train_dataset,indices3)
            
            train_loader0=DataLoader(train_dataset0, batch_size=args.batch_size, shuffle=False) 
            train_loader1=DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=False)    
            train_loader2=DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=False)        
            train_loader3=DataLoader(train_dataset3, batch_size=args.batch_size, shuffle=False)
            
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model1.train()
            
            
            lr_scheduler.step()

            step1 = 0
            step2 = 0
            step3 = 0
            for batch_data in train_loader0:
                step_start = time.time()
                
                inputs, masks = (
                    batch_data["image"].to(device),
                    batch_data["mask"].to(device),
                )
                optimizer1.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model1(inputs)
                    loss = loss_function(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer1)
                scaler.update()
           
                # print(
                    # f"{step1}/{len(train_dataset1) // train_loader1.batch_size}"
                    # f", train_loss: {loss.item():.4f}"
                    # f", step time: {(time.time() - step_start):.4f}"
                # )
                
            model1.eval()
            with torch.no_grad():
                step1 += 1

                for val_data in train_loader1:
                    val_inputs, val_masks = (
                        val_data["image"].to(device),
                        val_data["mask"].to(device),
                    )
                    val_outputs = inference1(val_inputs)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_masks)
                    dice_metric_batch(y_pred=val_outputs, y=val_masks)

                metric11 = dice_metric.aggregate().item()
                metric_values1.append(metric11)
                metric_batch1 = dice_metric_batch.aggregate()
                metric_tc1 = metric_batch1[0].item()
                metric_values_tc1.append(metric_tc1)
                metric_wt1 = metric_batch1[1].item()
                metric_values_wt1.append(metric_wt1)
                metric_et1 = metric_batch1[2].item()
                metric_values_et1.append(metric_et1)
                dice_metric.reset()
                dice_metric_batch.reset()

                    
                print(
                    f"model 1 current dice on train sample 1: {metric11:.4f}"
                    f" tc: {metric_tc1:.4f} wt: {metric_wt1:.4f} et: {metric_et1:.4f}"                   

                )              
                if metric11 > best_metric11: # In case we want to use this later
                    best_metric11 = metric11
                    
                for val_data in train_loader2:
                    val_inputs, val_masks = (
                        val_data["image"].to(device),
                        val_data["mask"].to(device),
                    )
                    val_outputs = inference1(val_inputs)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_masks)
                    dice_metric_batch(y_pred=val_outputs, y=val_masks)

                metric12 = dice_metric.aggregate().item()
                metric_values1.append(metric12)
                metric_batch1 = dice_metric_batch.aggregate()
                metric_tc1 = metric_batch1[0].item()
                metric_values_tc1.append(metric_tc1)
                metric_wt1 = metric_batch1[1].item()
                metric_values_wt1.append(metric_wt1)
                metric_et1 = metric_batch1[2].item()
                metric_values_et1.append(metric_et1)
                dice_metric.reset()
                dice_metric_batch.reset()

                    
                print(
                    f"model 1 current dice on train sample 2: {metric12:.4f}"
                    f" tc: {metric_tc1:.4f} wt: {metric_wt1:.4f} et: {metric_et1:.4f}"                   

                )              
                
                if metric12 > best_metric12: # In case we want to use this later
                    best_metric12 = metric12



                for val_data in train_loader3:
                    val_inputs, val_masks = (
                        val_data["image"].to(device),
                        val_data["mask"].to(device),
                    )
                    val_outputs = inference1(val_inputs)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_masks)
                    dice_metric_batch(y_pred=val_outputs, y=val_masks)

                metric13 = dice_metric.aggregate().item()
                metric_values1.append(metric13)
                metric_batch1 = dice_metric_batch.aggregate()
                metric_tc1 = metric_batch1[0].item()
                metric_values_tc1.append(metric_tc1)
                metric_wt1 = metric_batch1[1].item()
                metric_values_wt1.append(metric_wt1)
                metric_et1 = metric_batch1[2].item()
                metric_values_et1.append(metric_et1)
                dice_metric.reset()
                dice_metric_batch.reset()

                    
                print(
                    f"model 1 current dice on train sample 3: {metric13:.4f}"
                    f" tc: {metric_tc1:.4f} wt: {metric_wt1:.4f} et: {metric_et1:.4f}"                   

                )              
                
                if metric13 > best_metric13: # In case we want to use this later
                    best_metric13 = metric13
                    best_metric_epoch1 = epoch + 1          
              
               
            
            metric1=metric11
            metric2=metric12            
            metric3=metric13
            
            if metric1>metric2:
                if metric1>metric3: 
                    # print("1 was best with an avg score of : ",metric1, " 2 & 3 :",metric2,metric3)
                   
                    if metric2>metric3: # 1>2>3
                        print("Adding 3")
                        indices0= np.concatenate((indices3,indices0))  
                        indices3=all_indices[max_index:max_index+bunch]
                        max_index+=bunch
                        metric+=metric3/step1
                    else: #1>3>2
                        print("Adding 2")
                        indices0= np.concatenate((indices2,indices0))  
                        indices2=all_indices[max_index:max_index+bunch]                  
                        max_index+=bunch
                        metric+=metric2/step1
                        
                else: # 3>1>2
                    print("Adding  2")
                    indices0= np.concatenate((indices2,indices0))  
                    indices2=all_indices[max_index:max_index+bunch]                  
                    max_index+=bunch
                    
                    # print("3 was best with an avg score of : ",metric3, "1 & 2 :",metric1,metric2)
                    metric+=metric2/step1
                    
            else:
                if metric2>metric3:
                    # print("2 was best with an avg score of : ",metric2, "1 & 3 :",metric1,metric3)
                    
                    
                    
                    if metric1>metric3: #2>1>3
                        print("Adding 3")
                        indices0= np.concatenate((indices3,indices0))  
                        indices3=all_indices[max_index:max_index+bunch]
                        max_index+=bunch
                        metric+=metric3/step1
                        
                    else: # 2>3>1
                        print("Adding 1")
                        indices0= np.concatenate((indices1,indices0))  
                        indices1=all_indices[max_index:max_index+bunch]
                        max_index+=bunch
                        metric+=metric1/step1
                    
                elif metric3>metric2: #3>2>1
                    print("Adding 1")
                    indices0= np.concatenate((indices1,indices0))  
                    indices1=all_indices[max_index:max_index+bunch]
                    max_index+=bunch
                    # print("3 was best with an avg score of : ",metric3, "1 & 2 :",metric1,metric2)
                    metric+=metric1/step1
                    
                print(f"time consumption of look {step1} is: {(time.time() - look_start):.4f}")
                    
        print(f"The best lowest dice score for {epoch+1} epoch was {metric}")  
        
        if metric>best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(
                        model1.state_dict(),
                        os.path.join(root_dir,"MBISone"+ date.today().isoformat()+'T'+str(datetime.today().hour)+'b'+ str(args.bunch)+"ms"+str(args.max_samples)+"e"+str(best_metric_epoch)))
        print(f"The best metric so far is {best_metric} at {best_metric_epoch}")    
        print(f"time consumption of look {step1} is: {(time.time() - epoch_start):.4f}")
        print("added samples: ",indices0[bunch:])
        
    total_time = time.time() - total_start

    # print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

