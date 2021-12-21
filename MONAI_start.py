# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 22:59:47 2021

@author: Ganesh
"""

import monai
import os
from monai.data import Dataset
from torchvision.io import read_image
import numpy as np
from monai.transforms import LoadImageD, EnsureChannelFirstD, AddChannelD,\
    ScaleIntensityD, ToTensorD, Compose, AsDiscreteD, SpacingD, OrientationD,\
    ResizeD, RandAffineD,EnsureType, Activations, AsDiscrete, MapTransform
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
)
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image
from monai.data import decollate_batch, list_data_collate
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import DataLoader
import re

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npy', '.gz'
]

# source: Nvidia HDGAN
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# makes a list of all image paths inside a directory
def make_dataset(dir):
    all_files = []
    images=[]
    masks=[]
    im_temp=[]
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    
    for root, fol, _ in sorted(os.walk(dir)): # list folders and root
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

# source: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert masks to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d
            
class BratsDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir=data_dir       
        self.transform=transform
        
    def __len__(self):
#         return len(os.listdir(self.mask_dir))
        return 100
    
    def __getitem__(self,idx):
        image=make_dataset(self.data_dir)[0][idx]
    
        mask=make_dataset(self.data_dir)[1][idx]
 

            
        item_dict={"image":image,"mask":mask}
        
        if self.transform:
            item_dict={"image":image,"mask": mask}
            item_dict=self.transform(item_dict)
            
        
        return item_dict
    
    
    
KEYS=("image","mask")

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
        RandSpatialCropd(keys=["image", "mask"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        EnsureTyped(keys=["image", "mask"]),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "mask"]),
    ]
)

#dataset=DecathlonDataset(root_dir="./", task="Task05_Prostate",section="training", transform=xform, download=True)
train_dataset=BratsDataset("../RSNA_ASNR_MICCAI_BraTS2021_TrainingData"  ,transform=train_transform )
val_dataset=BratsDataset("../RSNA_ASNR_MICCAI_BraTS2021_ValidationData",transform=val_transform)
train_loader=DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=1, shuffle=True)

# pick one image from DecathlonDataset to visualize and check the 4 channels
print(f"image shape: {val_dataset[2]['image'].shape}")
plt.figure("image", (24, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.title(f"image channel {i}")
    plt.imshow(val_dataset[2]["image"][i, :, :, 60].detach().cpu(), cmap="gray")
plt.show()
# also visualize the 3 channels mask corresponding to this image
print(f"mask shape: {val_dataset[2]['mask'].shape}")
plt.figure("mask", (18, 6))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title(f"mask channel {i}")
    plt.imshow(val_dataset[2]["mask"][i, :, :, 60].detach().cpu())
plt.show()

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