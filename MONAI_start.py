from typing import List, Optional, Sequence, Tuple, Union

import pandas
# print(pandas.__version__)
import nibabel

import os
import monai
from monai.data import Dataset
from monai.utils import set_determinism

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

from monai.networks.nets import SegResNet
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import DataLoader
import numpy as np
from datetime import date, datetime
import sys
import re
import torch
import time
import argparse

parser=argparse.ArgumentParser(description="Monai Seg main")

parser.add_argument("--load_barlow",default =1, type=int,help="flag to use barlow twins backbone to initialise weight")
parser.add_argument("--load_save",default =1, type=int,help="flag to use saved model weight")
parser.add_argument("--load_path",default="./2022-01-20T16best_metric_model.pth", type=str, help="file path to load previously saved model")
parser.add_argument("--batch_size",default=8, type=int, help="to define batch size")
parser.add_argument("--save_name", default="Best_metric_model.pth",type=str, help="save name")
parser.add_argument("--upsample", default="DECONV",type=str, help="flag to choose deconv options- NONTRAINABLE, DECONV, PIXELSHUFFLE")
parser.add_argument("--barlow_final",default=1, type=int, help="flag to use checkpoint instead of final model for barlow")
parser.add_argument("--bar_model_name",default="checkpoint.pth", type=str,help="model name to load")
parser.add_argument("--max_samples",default=10000,type=int,help="max number of samples to use for training")

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
            
class BratsDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        self.image_list=make_dataset(data_dir)[0]  
        self.mask_list=make_dataset(data_dir)[1] 
        self.transform=transform
        
    def __len__(self):
#         return len(os.listdir(self.mask_dir))
        return min(args.max_samples,len(self.mask_list))
    
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
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        RandGaussianNoised(keys="image"),
        RandRotateD(keys=["image","mask"],range_x=0.1,range_y=0.1, range_z=0.1,prob=0.5),
        RandGaussianSmoothD(keys="image",sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5), prob=0.1),
        RandBiasFieldD(keys="image"),        
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

#dataset=DecathlonDataset(root_dir="./", task="Task05_Prostate",section="training", transform=xform, download=True)
train_dataset=BratsDataset("./RSNA_ASNR_MICCAI_BraTS2021_TrainingData"  ,transform=train_transform )
val_dataset=BratsDataset("./RSNA_ASNR_MICCAI_BraTS2021_ValidationData",transform=val_transform)
train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
print("All Datasets assigned")

##################~~~~~~~~~~~~~~~~~~~~~~~~~Model Definition~~~~~~~~~~~~~~~~~~~~~~~#################
root_dir="./"

max_epochs = 100
val_interval = 1
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=32,
    norm="instance",
    in_channels=4,
    out_channels=3,
    upsample_mode=UpsampleMode[args.upsample]
    
).to(device)

model=torch.nn.DataParallel(model)
print("Model defined and passed to GPU")

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)
if args.load_barlow==1:
    # Load model weight from Barlow Twins
    ckpt=torch.load("./barlowtwins/checkpoint/"+args.bar_model_name)
    if args.barlow_final==0:

        # print(ckpt.keys())

        for k in model.state_dict().keys():
            # print(ckpt["model"].keys())
            ckpt["model"][k]=ckpt["model"].pop("module.backbone."+k[7:])
        
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt,strict=False)

    print("Model weights loaded from pretrained Barlow Twins Backbone")

if args.load_save==1:    
    ckpt=torch.load(args.load_path)
    
    model.load_state_dict(ckpt, strict=False)
    print("Model weights loaded from best metric model")
    

# define inference method
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
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

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
                torch.save(
                    model.state_dict(),
                    os.path.join(root_dir, date.today().isoformat()+'T'+str(datetime.today().hour)+ args.save_name),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

# ####~~~~~~~~~~~~~~~~PLOTS~~~~~~~~~~~~~~~~
# plt.figure("train", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Epoch Average Loss")
# x = [i + 1 for i in range(len(epoch_loss_values))]
# y = epoch_loss_values
# plt.xmask("epoch")
# plt.plot(x, y, color="red")
# plt.subplot(1, 2, 2)
# plt.title("Val Mean Dice")
# x = [val_interval * (i + 1) for i in range(len(metric_values))]
# y = metric_values
# plt.xmask("epoch")
# plt.plot(x, y, color="green")
# plt.show()

# plt.figure("train", (18, 6))
# plt.subplot(1, 3, 1)
# plt.title("Val Mean Dice TC")
# x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
# y = metric_values_tc
# plt.xmask("epoch")
# plt.plot(x, y, color="blue")
# plt.subplot(1, 3, 2)
# plt.title("Val Mean Dice WT")
# x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
# y = metric_values_wt
# plt.xmask("epoch")
# plt.plot(x, y, color="brown")
# plt.subplot(1, 3, 3)
# plt.title("Val Mean Dice ET")
# x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
# y = metric_values_et
# plt.xmask("epoch")
# plt.plot(x, y, color="purple")
# plt.show()

################## ~~~~~~~~~~~~~Best model output with image and mask ~~~~~~~~~~ #######################
# model.load_state_dict(
#     torch.load(os.path.join(root_dir, "best_metric_model.pth"))
# )
# model.eval()
# with torch.no_grad():
#     # select one image to evaluate and visualize the model output
#     val_input = val_dataset[6]["image"].unsqueeze(0).to(device)
#     roi_size = (128, 128, 64)
#     sw_batch_size = 4
#     val_output = inference(val_input)
#     val_output = post_trans(val_output[0])
#     plt.figure("image", (24, 6))
#     for i in range(4):
#         plt.subplot(1, 4, i + 1)
#         plt.title(f"image channel {i}")
#         plt.imshow(val_dataset[6]["image"][i, :, :, 70].detach().cpu(), cmap="gray")
#     plt.show()
#     # visualize the 3 channels mask corresponding to this image
#     plt.figure("mask", (18, 6))
#     for i in range(3):
#         plt.subplot(1, 3, i + 1)
#         plt.title(f"mask channel {i}")
#         plt.imshow(val_dataset[6]["mask"][i, :, :, 70].detach().cpu())
#     plt.show()
#     # visualize the 3 channels model output corresponding to this image
#     plt.figure("output", (18, 6))
#     for i in range(3):
#         plt.subplot(1, 3, i + 1)
#         plt.title(f"output channel {i}")
#         plt.imshow(val_output[i, :, :, 70].detach().cpu())
#     plt.show()

########### ~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~FIX this

# test_transforms0 = Compose(
    # [
        # LoadImaged(keys=["image", "mask"]),
        # EnsureChannelFirstd(keys=["image"]),
        # ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        # Orientationd(keys=["image"], axcodes="RAS"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # EnsureTyped(keys=["image", "mask"]),
    # ]
# ) #>>>>>>>>>>>>>>   TRANSFORMS DEFINED 

# test_transforms1 = Compose(
    # [
        # LoadImaged(keys=["image", "mask"]),
        # EnsureChannelFirstd(keys=["image"]),
        # ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        
        # Orientationd(keys=["image"], axcodes="RAS"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RandFlipd(keys=["image", "mask"], prob=1, spatial_axis=0),
        # EnsureTyped(keys=["image", "mask"]),
    # ]
# )
# at1=RandFlip(prob=1,spatial_axis=0)

# test_transforms2 = Compose(
    # [
        # LoadImaged(keys=["image", "mask"]),
        # EnsureChannelFirstd(keys=["image"]),
        # ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        
        # Orientationd(keys=["image"], axcodes="RAS"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RandFlipd(keys=["image", "mask"], prob=1, spatial_axis=1),
        # EnsureTyped(keys=["image", "mask"]),
    # ]
# )
# at2=RandFlip(prob=1,spatial_axis=1)

# test_transforms3 = Compose(
    # [
        # LoadImaged(keys=["image", "mask"]),
        # EnsureChannelFirstd(keys=["image"]),
        # ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        
        # Orientationd(keys=["image"], axcodes="RAS"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RandFlipd(keys=["image", "mask"], prob=1, spatial_axis=2),
        # EnsureTyped(keys=["image", "mask"]),
    # ]
# )
# at3=RandFlip(prob=1,spatial_axis=2)

# test_transforms4 = Compose(
    # [
        # LoadImaged(keys=["image", "mask"]),
        # EnsureChannelFirstd(keys=["image"]),
        # ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        
        # Orientationd(keys=["image"], axcodes="RAS"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RandGaussianNoised(keys="image"),
        # EnsureTyped(keys=["image", "mask"]),
    # ]
# )

# test_transforms5 = Compose(
    # [
        # LoadImaged(keys=["image", "mask"]),
        # EnsureChannelFirstd(keys=["image"]),
        # ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),  
        # Orientationd(keys=["image"], axcodes="RAS"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RotateD(keys=["image","mask"],angle=[0.1,0,0]),
        # EnsureTyped(keys=["image", "mask"]),
    # ]
# )

# at5=Rotate(angle=[-0.1,0,0])

# test_transforms6 = Compose(
    # [
        # LoadImaged(keys=["image", "mask"]),
        # EnsureChannelFirstd(keys=["image"]),
        # ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),  
        # Orientationd(keys=["image"], axcodes="RAS"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RandGaussianSmoothD(keys="image",sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5), prob=1),
        # EnsureTyped(keys=["image", "mask"]),
    # ]
# )

# test_transforms7 = Compose(
    # [
        # LoadImaged(keys=["image", "mask"]),
        # EnsureChannelFirstd(keys=["image"]),
        # ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),  
        # Orientationd(keys=["image"], axcodes="RAS"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RandBiasFieldD(keys="image",prob=1), 
        # EnsureTyped(keys=["image", "mask"]),
    # ]
# )

# test_transforms8 = Compose(
    # [
        # LoadImaged(keys=["image", "mask"]),
        # EnsureChannelFirstd(keys=["image"]),
        # ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),  
        # Orientationd(keys=["image"], axcodes="RAS"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RandScaleIntensityd(keys="image", factors=0.1, prob=1), 
        # EnsureTyped(keys=["image", "mask"]),
    # ]
# )
# test_transforms9 = Compose(
    # [
        # LoadImaged(keys=["image", "mask"]),
        # EnsureChannelFirstd(keys=["image"]),
        # ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),  
        # Orientationd(keys=["image"], axcodes="RAS"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RandShiftIntensityd(keys="image", offsets=0.1, prob=1),
        # EnsureTyped(keys=["image", "mask"]),
    # ]
# )

# all_transforms=[test_transforms0,test_transforms1,test_transforms2,test_transforms3,test_transforms4,test_transforms5,test_transforms6,test_transforms7,test_transforms8,test_transforms9]

# class TestDataset(Dataset):
    # def __init__(self,data_dir,transform=None):
        # self.image_list=make_dataset(data_dir)[0]  
        # self.mask_list=make_dataset(data_dir)[1] 
        # self.transform=transform
        
    # def __len__(self):
        # return len(os.listdir(self.mask_dir))
        # return min(args.max_samples,len(self.mask_list))
    
    # def __getitem__(self,idx):
        # image=self.image_list[idx]
    
        # mask=self.mask_list[idx] 

            
        # item_dict={"image":image,"mask":mask}
        
        # test_list=[]
        # if self.transform:
            # item_dict={"image":image,"mask": mask}
            # for transform in self.transform:
                # item_dict=transform(item_dict)
                # test_list.append item_dict
            
        
        # return test_list

# test_ds=TestDataset("./RSNA_ASNR_MICCAI_BraTS2021_TestData",transform=all_transforms,shuffle=True)



# test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4) # this should return 10 different instances



# post_transforms = Compose([
    # EnsureTyped(keys="pred"),
    # Invertd(
        # keys="pred",
        # transform=test_transforms,
        # orig_keys="image",
        # meta_keys="pred_meta_dict",
        # orig_meta_keys="image_meta_dict",
        # meta_key_postfix="meta_dict",
        # nearest_interp=False,
        # to_tensor=True,
    # ),
    # Activationsd(keys="pred", sigmoid=True),
    # AsDiscreted(keys="pred", threshold=0.5),
# ])

# model.load_state_dict(torch.load(
    # os.path.join(root_dir, "best_metric_model.pth")))
# model.eval()

# with torch.no_grad():
    # for test_data in test_loader:
        # for c, each_dict in enumerate(test_data):
            # test_inputs = each_dict["image"].to(device)
            # each_dict["pred"] = inference(test_inputs)
            # each_dict = [post_transforms(i) for i in decollate_batch(each_dict)]
            # test_outputs, test_masks = from_engine(["pred", "mask"])(each_dict)
            
            # if c==0:
                # collector=test_outputs[None,:]
            
            # if c==1:
                # test_outputs=at1(test_outputs)
                # test_masks =at1(test_masks)
            # if c==2:
                # test_outputs=at2(test_outputs)
                # test_masks =at2(test_masks)
            # if c=3:
                # test_outputs=at3(test_outputs)
                # test_masks =at3(test_masks)
            # if c=5:
                # test_outputs=at5(test_outputs)
                # test_masks =at5(test_masks)
            # collector.cat((collector,test_outputs[None,:])) # this should collect all predictions
            
            
            
            
            ## torch.
        
        # mean_pred=collector.mean(dim=0, keepdim=True).flatten(0,1) # get the same dim as mask
        # dice_metric(y_pred=mean_pred, y=test_masks)
        # dice_metric_batch(y_pred=test_outputs, y=test_masks)

    # metric_org = dice_metric.aggregate().item()
    # metric_batch_org = dice_metric_batch.aggregate()

    # dice_metric.reset()
    # dice_metric_batch.reset()

# metric_tc, metric_wt, metric_et = metric_batch[0].item(), metric_batch[1].item(), metric_batch[2].item()

# print("Metric on original image spacing: ", metric)
# print(f"metric_tc: {metric_tc:.4f}")
# print(f"metric_wt: {metric_wt:.4f}")
# print(f"metric_et: {metric_et:.4f}")


######################~~~~~~~~3D Seg End ~~~~~~###################



# device=torch.device(("cuda:0"))
# net=UNet(dimensions=2, in_channels=1, out_channels=3, channels=(16,32,64,128,256),strides=(2,2,2,2),num_res_units=2
#         ,norm=Norm.BATCH).to(device)
# loss_function=DiceLoss(to_onehot_y=True, softmax=True)
# opt=torch.optim.Adam(net.parameters(),1e-2)

# dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
# post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]) # learn this after model run
# val_interval=2 # This is used as epoch counter to run validation later
# best_metric=-1 # This is initialised such that tbe best mnetric is saved for iteration later
# best_metric_epoch=-1 # same as best metric but for epoch
# epoch_loss_values=list() # To store loss values
# metric_values=list() # to store metric values later
# writer=SummaryWriter()
# for epoch in range(10): # run for the number of epochs you want to run
#     print("-"*10)
#     print(f"epoch{epoch+1}/{10}")
#     net.train()
#     epoch_loss= 0 # to store loss presumably
#     step = 0 # to maintain counter for total number of batches
#     for batch_data in train_loader:  # cycle through the dataset one batch at a time
#         step+=1 # add 1 for each batch
#         inputs,masks= torch.mean(batch_data["image"],1,True).to(device),torch.mean(batch_data["mask"],1,True).to(device)
#         #since data incorrectly saved with 4 channels take mean along the channel axis
#         opt.zero_grad()
#         outputs=net(inputs) # forward pass of the model to give you output of the final conv block
#         loss=loss_function(outputs,masks) # compare output and GT to calculate loss
#         loss.backward # Back propagate calculated loss using chain rule
#         opt.step() # Step each weight based on its specific contribution to the loss
#         epoch_loss += loss.item() # cumulative sum of losses through the epoch
#         epoch_len = len(train_dataset) // train_loader.batch_size # how many batches in a epoch
#         print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}") # print which batch out of total batches
#         writer.add_scalar("train_loss",loss.item(),epoch_len * epoch + step)
#     epoch_loss /= step # average loss at each batch in a given epoch
#     epoch_loss_values.append(epoch_loss) # adding to the loss list for plotting later
#     print(f"epoch {epoch+1} average loss: {epoch_loss: .4f}")
    
#     if (epoch+1)% val_interval ==0: # ie run only at every x eppochs where x=val_interval
#         net.eval()
#         with torch.no_grad():
#             val_images=None # initialising so you don't get an error? Can i remove this
#             val_masks= None
#             val_outputs= None
#             for val_data in val_loader:
#                 val_images, val_masks= torch.mean(val_data["image"],1, True).to(device), torch.mean(val_data["mask"],1,True).to(device)
#                 roi_size=(96,96) # load images in patches using ROI. Interesting how this is only called at eval
#                 sw_batch_size =4 # sliding window batch size
#                 val_outputs = sliding_window_inference(val_images,roi_size,sw_batch_size, net) # Learn later
#                 val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)] # learn after first successful run
                
#                 # compute metric for current iteration
#                 dice_metric(y_pred=val_outputs, y=val_masks)
            
#             #aggregate the final mean dice result
#             metric= dice_metric.aggregate().item()
#             #reset the status for next validation round
#             dice_metric.reset()
            
#             metric_values.append(metric)
#             if metric > best_metric:
#                 best_metric = metric
#                 best_metric_epoch = epoch + 1
#                 torch.save(net.state_dict(), "best_metric_model_segmentation2d_dict.pth")
#                 print("saved new best metric model")
#             print(
#                 "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
#                     epoch + 1, metric, best_metric, best_metric_epoch
#                 )
#             )
#             writer.add_scalar("val_mean_dice", metric, epoch + 1)
#             # plot the last model output as GIF image in TensorBoard with the corresponding image and mask
#             plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
#             plot_2d_or_3d_image(val_masks, epoch + 1, writer, index=0, tag="mask")
#             plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

#     print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
#     writer.close()  