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
    SpatialResampled,
    SpatialResample,
    RandAffined,
    CropForegroundd,
    RandGaussianSmoothd,
    
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
from glob import glob


     
        
if __name__=="__main__":

    parser=argparse.ArgumentParser(description="Monai Seg main")

    parser.add_argument("--lr",default=2e-4,type=float,help="learning rate")
    parser.add_argument("--model",default="SegResNet",type=str,help="name of model to use")
    parser.add_argument("--load_save",default =0, type=int,help="flag to use saved model weight")
    parser.add_argument("--load_path",default="/scratch/a.bip5/BraTS 2021/2022-01-20T16best_metric_model.pth", type=str, help="file path to load previously saved model")
    parser.add_argument("--batch_size",default=1, type=int, help="to define batch size")
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
    parser.add_argument("--T_max",default=20,type=int,help="scheduling param")
    parser.add_argument("--workers",default=8,type=int,help="number of workers(cpu threads)")
    args=parser.parse_args()

    print(' '.join(sys.argv))
    
    os.environ['PYTHONHASHSEED']=str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    set_determinism(seed=args.seed)

    IMG_EXTENSIONS = [
         '.gz'
    ]##'.jpg', '.JPG', '.jpeg', '.JPEG',
       ## '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npy',


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
        for root, fol, fnames in os.walk(data_dir+'/rawdata'):
            for f in fnames:                      
                fpath = os.path.join(root, f)
                if f.endswith('flair.nii.gz'):                
                    pass
                elif f.endswith('.nii.gz'):
                    im_temp.append(fpath)         
            if im_temp:
                images.append(im_temp)                   
            im_temp = []
        masks= sorted(glob(f"{data_dir+'/derivatives'}/**/*.nii.gz",recursive=True))                        
        return sorted(images), masks
        


    # A source: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
    class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
        """
        Convert masks to multi channels based on brats classes:
        mask 2 is the peritumoral edema
        mask 4 is the GD-enhancing tumor
        mask 1 is the necrotic and non-enhancing tumor core
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor).

        """

        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                result = []
                # merge mask 1 and mask 4 to construct TC
                result.append(np.logical_or(d[key] == 4, d[key] == 1))
                # merge masks 1, 2 and 4 to construct WT
                result.append(
                    np.logical_or(
                        np.logical_or(d[key] == 2, d[key] == 4), d[key] == 1
                    )
                )
                # mask 4 is ET
                result.append(d[key] == 4)
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
                train_indices=np.delete(indexes,val_indices)#indexes[(i-5)*fold:(i-4)*fold]
                
               
    class BratsDataset(Dataset):
        def __init__(self,data_dir,transform=None):            
            self.image_list=make_dataset(data_dir)[0]            
            self.mask_list=make_dataset(data_dir)[1] 
            self.transform=transform   
            
        def __len__(self):
            return min(args.max_samples,len(self.mask_list))#        
        def __getitem__(self,idx):              
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
            LoadImaged(keys=["image", "mask"]),
            AddChannelD(keys="mask"),
            EnsureChannelFirstD(keys="image"),
            
            CropForegroundd(["image", "mask"], source_key="image"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
           
            SpacingD(
                keys=["image", "mask"],
                pixdim=(2.0, 2.0, 2.0),
                mode="bilinear",
            ),   
            OrientationD(keys=["image", "mask"],axcodes="RAS"),
            RandSpatialCropd(
            ["image", "mask"], roi_size=(64,64,64), random_size=False
            ),            
            RandAffined(
            ["image", "mask"],
            prob=0.15,
            spatial_size=(64,64,64),
            rotate_range=[30 * np.pi / 180] * 3,
            scale_range=[0.3] * 3,
            mode=("bilinear", "bilinear"),            
            ),                      
            RandRotateD(keys=["image","mask"],range_x=0.1,range_y=0.1, range_z=0.1,prob=0.5),  
            RandFlipd(["image", "mask"], prob=0.5, spatial_axis=0),
            RandFlipd(["image", "mask"], prob=0.5, spatial_axis=1),
            RandFlipd(["image", "mask"], prob=0.5, spatial_axis=2),
            RandGaussianNoised("image", prob=0.15, std=0.1),
            RandGaussianSmoothd(
            "image",
            prob=0.15,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
            ),
            
            
            
            RandScaleIntensityd(keys="image", factors=0.3, prob=0.15),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
            AsDiscreted("mask", threshold=0.5),
            EnsureTyped(keys=["image", "mask"]),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstD(keys="image"),
            AddChannelD(keys="mask"),            
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),            
            EnsureTyped(keys=["image", "mask"]),
        ]
    )

    #dataset=DecathlonDataset(root_dir="/scratch/a.bip5/BraTS 2021/", task="Task05_Prostate",section="training", transform=xform, download=True)
    train_dataset=BratsDataset("/scratch/a.bip5/BraTS 2021/dataset-ISLES22^public^unzipped^version"  ,transform=train_transform ) 
    val_dataset=BratsDataset("/scratch/a.bip5/BraTS 2021/dataset-ISLES22^public^unzipped^version"  ,transform=val_transform )

    
    print('len(train_dataset)',len(train_dataset))

    if args.CV_flag==1:
        print("loading cross val data")
        val_dataset=Subset(train_dataset,train_indices)
        train_dataset1=Subset(train_dataset,train_indices)
        
    else:     
        print("loading data for single model training")
        val_dataset1=Subset(val_dataset,np.arange(50))#np.arange(800,1000))
        train_dataset1=Subset(train_dataset,np.arange(50,250))#np.arange(800))

  
        
    # print("files  to be processed: ", train_dataset.files)
    train_loader=DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=True,num_workers=args.workers)
    val_loader=DataLoader(val_dataset1, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)
    print("All Datasets assigned")

    root_dir="/scratch/a.bip5/BraTS 2021/dataset-ISLES22^public^unzipped^version/"

    max_epochs = args.epochs
    val_interval = 1
    VAL_AMP = True

    # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    
    class CustomActivation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(0.3))
            self.b = torch.nn.Parameter(torch.tensor(0.3))
            self.c = torch.nn.Parameter(torch.tensor(0.3))

            self.register_parameter("a", self.a)
            self.register_parameter("b", self.b)
            self.register_parameter("c", self.c)

        def forward(self, x):
            return self.a * F.relu(x) + self.b * torch.sigmoid(x) + self.c * torch.tanh(x)


    class SegResNetWithCustomActivation(SegResNet):
        # initialize with these params
        def __init__(self, in_channels, out_channels,blocks_down,blocks_up,init_filters,upsample_mode,norm,spatial_dims=3,act='RELU',dropout_prob=None,norm_name='',use_conv_final=True,num_groups=8):
           # print(f"act argument: {act}")
           #initialize parent classes with these params instead of its original
            super().__init__(spatial_dims,init_filters,in_channels,out_channels,dropout_prob,act,norm,norm_name,num_groups,use_conv_final,blocks_down,blocks_up,upsample_mode)
            self.total_blocks=22
            self.activations = [CustomActivation() for i in range(self.total_blocks)]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.convInit(x)
            if self.dropout_prob is not None:
                x = self.dropout(x)

            down_x = []

            for down in self.down_layers:
                x = down(x)
                x = self.activations[0](x)
                down_x.append(x)
                
                
            x = self.activations[0](x)

            down_x.reverse()
            activations_idx = 1
            for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
                x = up(x) + down_x[i + 1]
                x = upl(x)
                
                
                x = self.activations[activations_idx](x)
                activations_idx += 1

            if self.use_conv_final:
                x = self.conv_final(x)

            return x

    torch.manual_seed(args.seed)
    model = SegResNetWithCustomActivation(
            in_channels=2,
            out_channels=1,
            blocks_down=[2, 4, 4, 4,4],
            blocks_up=[1, 1, 1,1],
            init_filters=1,            
            norm='instance',
            upsample_mode=UpsampleMode[args.upsample],      
                        
        ).to(device)

    # sys.exit(0)
    
    # torch.manual_seed(args.seed)
    # if args.model=="UNet":
         # model=UNet(
            # spatial_dims=3,
            # in_channels=4,
            # out_channels=3,
            # channels=(64,128,256,512,1024),
            # strides=(2,2,2,2)
            # ).to(device)
    # elif args.model=="SegResNet":
        # model = SegResNet(
            # blocks_down=[2, 4, 4, 4,4],
            # blocks_up=[1, 1, 1,1],
            # init_filters=3,
            # norm="instance",
            # in_channels=2,
            # out_channels=1,
            # upsample_mode=UpsampleMode[args.upsample]    
            # ).to(device)

    # else:
        # model = locals() [args.model](4,3).to(device)

    with torch.cuda.amp.autocast():
        summary(model,(2,64,64,64))

    model=torch.nn.DataParallel(model)
    print("Model defined and passed to GPU")
    
    if args.load_save==1:
        model.load_state_dict(torch.load("/scratch/a.bip5/BraTS 2021/"+args.load_path),strict=False)
        print("loaded saved model ", args.load_path)

    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)

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
                loss = loss_function(outputs, masks)
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
                
        print('lr_scheduler.get_last_lr() = ',lr_scheduler.get_last_lr())
        if epoch>99:
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
               
               
                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    if args.CV_flag==1:
                        if epoch>99:
                            torch.save(
                                model.state_dict(),
                                os.path.join(root_dir, args.model+"CV"+str(args.fold_num)+"ms"+str(args.max_samples)+"rs"+str(args.seed)+args.method+'extra'))
                        else:
                        
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
                    
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                

                
        print(f"time consumption of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
    with open ('./time_consumption.csv', 'a') as sample:
        sample.write(f"{args.model},{args.method},{total_time},{date.today().isoformat()},{args.fold_num},{args.CV_flag},{args.seed},{args.epochs}\n")
       
