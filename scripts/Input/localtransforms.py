import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')
from monai.transforms import (EnsureChannelFirstD, ToMetaTensorD,\
    ToTensorD,
    CropForegroundd,
    ScaleIntensityD, SpacingD, OrientationD,\
    ResizeD, RandAffineD,
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    CastToTyped,
    Invertd,
    RandAffined,
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
    CenterSpatialCropd,
    SpatialPadd,
    RemoveSmallObjectsd,
    RandGaussianSmoothd,
    RandSpatialCropd,   
    EnsureTyped,
    EnsureType,
    KeepLargestConnectedComponentd
)
import random
from Input.config import roi, base_transform_probability
import torch
import numpy as np
from monai.data import MetaTensor
from scipy.ndimage import center_of_mass

class Flipper(MapTransform):
    def __init__(self, keys, mask_key):
        super().__init__(keys)
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]

            # Ensure image and mask are MetaTensors
            if not isinstance(image, MetaTensor):
                raise TypeError(f"Expected image to be a MetaTensor but got {type(image)}.")

            # Extract metadata from the original MetaTensor for image
            image_meta = image.meta

            # Perform flipping and concatenation operations on numpy array for image
            image_np = image.numpy()

            # Initialize a list to store transformed channels for image
            transformed_image_channels = []

            # Flip and concatenate each channel for image
            for i in range(image_np.shape[0]):
                single_channel = image_np[i, ...]

                # Flip the channel in the sagittal axis
                flipped_channel = np.flip(single_channel, axis=1)

                # Concatenate the original and flipped channels
                concatenated_channel = np.concatenate((single_channel[np.newaxis, ...], flipped_channel[np.newaxis, ...]), axis=0)

                # Add the concatenated channel to the list
                transformed_image_channels.append(concatenated_channel)

            # Stack all transformed channels and convert back to MetaTensor for image
            transformed_image_np = np.concatenate(transformed_image_channels, axis=0)
            d[key] = MetaTensor(transformed_image_np, meta=image_meta)

            ## Also process the mask
            # mask = d[self.mask_key]

            ## Extract metadata from the original MetaTensor for mask
            # mask_meta = mask.meta

            ## Perform flipping and concatenation operations on numpy array for mask
            # mask_np = mask.numpy()

            ## Initialize a list to store transformed channels for mask
            # transformed_mask_channels = []

            ## Flip and concatenate each channel for mask
            # for i in range(mask_np.shape[0]):
                # single_channel = mask_np[i, ...]

                ## Flip the channel in the sagittal axis
                # flipped_channel = np.flip(single_channel, axis=1)

                ## Concatenate the original and flipped channels
                # concatenated_channel = np.concatenate((single_channel[np.newaxis, ...], flipped_channel[np.newaxis, ...]), axis=0)

                ## Add the concatenated channel to the list
                # transformed_mask_channels.append(concatenated_channel)

            ## Stack all transformed channels and convert back to MetaTensor for mask
            # transformed_mask_np = np.concatenate(transformed_mask_channels, axis=0)
            # d[self.mask_key] = MetaTensor(transformed_mask_np, meta=mask_meta)

        return d
        
class PseudoAtlas(MapTransform):
    def __init__(self, keys, mask_key):
        super().__init__(keys)
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            mask = d[self.mask_key]

            # Ensure image and mask are MetaTensors
            if not isinstance(image, MetaTensor):
                raise TypeError(f"Expected image to be a MetaTensor but got {type(image)}.")
            if not isinstance(mask, MetaTensor):
                raise TypeError(f"Expected mask to be a MetaTensor but got {type(mask)}.")

            # Extract metadata from the original MetaTensor
            meta = image.meta

            # Perform operations on numpy array
            image_np = image.numpy()
            mask_np = mask.numpy()

            # Calculate the centroid of the lesion from the mask
            centroid = center_of_mass(mask_np)

            # Initialize a list to store transformed channels
            transformed_channels = []

            # Apply transformation to each channel
            for i in range(image_np.shape[0]):
                # Process each channel
                single_channel = image_np[i, ...]

                # Mask the hemisphere containing the lesion
                masked_channel = self.mask_hemisphere(single_channel, centroid)

                # Flip the channel in the sagittal axis
                flipped_channel = np.flip(masked_channel, axis=1)

                # Concatenate the original and flipped channels
                concatenated_channel = np.concatenate((single_channel[np.newaxis, ...], flipped_channel[np.newaxis, ...]), axis=0)

                # Add the concatenated channel to the list
                transformed_channels.append(concatenated_channel)

            # Stack all transformed channels and convert back to MetaTensor
            transformed_image_np = np.concatenate(transformed_channels, axis=0)
            d[key] = MetaTensor(transformed_image_np, meta=meta)

        return d
        
    def mask_hemisphere(self, image, centroid):
        # Mask the hemisphere based on the centroid
        y_midpoint = image.shape[1] / 2
        if centroid[1] > y_midpoint:
            # Lesion in the right hemisphere, mask the left
            image[:, :int(y_midpoint), :] = 0
        else:
            # Lesion in the left hemisphere, mask the right
            image[:, int(y_midpoint):, :] = 0
        return image

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
        
class ConvertToMultiChannelBasedOnBratsClassesd_val(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
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
            # merge label 1 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 3)
            d[key] = torch.stack(result, axis=0).float()
        return d
        
class ConvertToSingleChannel(MapTransform):
    """
    Convert labels to single channels based on brats classes:
    label 2 is the peritumoral edema
    label 3 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            tc,wt,et=d[key]
            result = torch.zeros_like(tc)
            # Binarize each tensor
            tc = (tc > 0).int()
            wt = (wt > 0).int()
            et = (et > 0)
            
            
            ed=torch.logical_and(torch.logical_not(tc),wt)
            ncr=torch.logical_and(torch.logical_not(et),tc)
            # merge label 1 and label 3 to construct TC
            result[ed]=2
            result[et]=3
            result[ncr]=1
            
            
            d[key] = result
        return d
            
KEYS=("image","mask")
print("Transforms not defined yet")

train_transform_atlas = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image","mask"],simple_keys=True),
        EnsureChannelFirstD(keys="image"),
       
        
        SpacingD(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        OrientationD(keys=["image", "mask"], axcodes="RAS"),
        RandSpatialCropd(keys=["image", "mask"], roi_size=roi, random_size=False),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandRotateD(keys=["image","mask"],range_x=0.1,range_y=0.1, range_z=0.1,prob=0.5),     
        
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),    
        
    ]
)

train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image","mask"],simple_keys=True),
        EnsureChannelFirstD(keys="image"),
        EnsureTyped(keys=["image", "mask"]),
        ConvertToMultiChannelBasedOnBratsClassesd_val(keys="mask"),
        SpacingD(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        OrientationD(keys=["image", "mask"], axcodes="RAS"),
        RandSpatialCropd(keys=["image", "mask"], roi_size=roi, random_size=False),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
       
        RandRotateD(keys=["image","mask"],range_x=0.1,range_y=0.1, range_z=0.1,prob=0.5),
       
        
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),    
        
    ]
)
train_transform_BP = Compose(
    [
        # load 2 Nifti images and stack them together
        LoadImaged(keys=["image","mask"]),
        EnsureChannelFirstD(keys=["image","mask"]),
        EnsureTyped(keys=["image", "mask"]),
        SpacingD(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        AsDiscreted(keys="mask",threshold=0.5),
        
        OrientationD(keys=["image", "mask"], axcodes="RAS"),
        RandSpatialCropd(keys=["image", "mask"], roi_size=roi, random_size=False),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
       
        RandRotateD(keys=["image","mask"],range_x=0.1,range_y=0.1, range_z=0.1,prob=0.5),
       
        
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),    
        
    ]
)
train_transform_CA = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image","mask"],simple_keys=True),
        EnsureChannelFirstD(keys="image"),
        EnsureTyped(keys=["image", "mask"]),
        ConvertToMultiChannelBasedOnBratsClassesd_val(keys="mask"),
        SpacingD(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        OrientationD(keys=["image", "mask"], axcodes="RAS"),
        RandSpatialCropd(keys=["image", "mask","map"], roi_size=roi, random_size=False),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandRotateD(keys=["image","mask","map"],range_x=0.1,range_y=0.1, range_z=0.1,prob=0.5),
       
        
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),    
        
    ]
)


val_transform_atlas = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstD(keys=["image","mask"]),
        # AddChannelD(keys="mask"), 
        SpacingD(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
        ),
        OrientationD(keys=["image", "mask"],axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  
        RandSpatialCropd(
        ["image", "mask"], roi_size=roi, random_size=False
        ),
        EnsureTyped(keys=["image", "mask"]),
    ]
)


train_transform_PA = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image","mask"],simple_keys=True),
        EnsureChannelFirstD(keys="image"),
        EnsureTyped(keys=["image", "mask"]),
        PseudoAtlas(keys="image", mask_key="mask"),
        ConvertToMultiChannelBasedOnBratsClassesd_val(keys="mask"),
        SpacingD(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        OrientationD(keys=["image", "mask"], axcodes="RAS"),
        RandSpatialCropd(keys=["image", "mask"], roi_size=roi, random_size=False),
       
        RandRotateD(keys=["image","mask"],range_x=0.1,range_y=0.1, range_z=0.1,prob=0.5),
       
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),    
        
    ]
)
train_transform_Flipper = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image","mask"],simple_keys=True),
        EnsureChannelFirstD(keys="image"),
        EnsureTyped(keys=["image", "mask"]),
        
        ConvertToMultiChannelBasedOnBratsClassesd_val(keys="mask"),
        Flipper(keys="image", mask_key="mask"),
        SpacingD(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        OrientationD(keys=["image", "mask"], axcodes="RAS"),
        RandSpatialCropd(keys=["image", "mask"], roi_size=roi, random_size=False),
       
        RandRotateD(keys=["image","mask"],range_x=0.1,range_y=0.1, range_z=0.1,prob=0.5),
       
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),    
        
    ]
)
val_transform_Flipper = Compose(
    [
        LoadImaged(keys=["image", "mask"],simple_keys=True),
        EnsureChannelFirstD(keys="image"),
        
        ConvertToMultiChannelBasedOnBratsClassesd_val(keys="mask"),
        Flipper(keys="image", mask_key="mask"),
        SpacingD(
            keys=["image","mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        OrientationD(keys=["image","mask"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # CenterSpatialCropd(keys=["image", "mask"], roi_size=[224,224,155]),
        # SpatialPadd(keys=["image", "mask"],spatial_size=[224,224,160],method='symmetric')
        # ToTensorD(keys=["image"], dtype=torch.float),
        ]
        
)
val_transform_PA = Compose(
    [
        LoadImaged(keys=["image", "mask"],simple_keys=True),
        EnsureChannelFirstD(keys="image"),
        PseudoAtlas(keys="image", mask_key="mask"),
        ConvertToMultiChannelBasedOnBratsClassesd_val(keys="mask"),
        
        SpacingD(
            keys=["image","mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        OrientationD(keys=["image","mask"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # CenterSpatialCropd(keys=["image", "mask"], roi_size=[224,224,155]),
        # SpatialPadd(keys=["image", "mask"],spatial_size=[224,224,160],method='symmetric')
        # ToTensorD(keys=["image"], dtype=torch.float),
        ]
        
)

train_transform_infuse = Compose(
    [
             
        RandSpatialCropd(keys=["image", "mask"], roi_size=roi, random_size=False),
       
        RandRotateD(keys=["image","mask"],range_x=0.1,range_y=0.1, range_z=0.1,prob=0.5),
       
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),    
        
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=["image", "mask"],simple_keys=True),
        EnsureChannelFirstD(keys="image"),
        
        ConvertToMultiChannelBasedOnBratsClassesd_val(keys="mask"),
        
        SpacingD(
            keys=["image","mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        OrientationD(keys=["image","mask"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # CenterSpatialCropd(keys=["image", "mask"], roi_size=[224,224,155]),
        # SpatialPadd(keys=["image", "mask"],spatial_size=[224,224,160],method='symmetric')
        # ToTensorD(keys=["image"], dtype=torch.float),
        ]
        
)


# mean_post_transforms = Compose(
    # [
        # EnsureTyped(keys=["pred"+str(i) for i in range(len(models))]), #gives pred0..pred1...
        # SplitChanneld(keys=["pred"+str(i) for i in range(10)]),
        
        # MeanEnsembled(
            # keys=["pred"+str(i) for i in range(len(models))], 
            # output_key="pred",
          #  # in this particular example, we use validation metrics as weights
          ## weights=wts,
        # ),
        # Activationsd(keys="pred", sigmoid=True),
        # AsDiscreted(keys="pred", threshold=0.2),
        # SaveImaged(keys="pred",output_dir='/scratch/a.bip5/BraTS 2021/ssensemblemodels0922/outputs', output_ext=".nii.gz",meta_key_postfix=None, output_postfix= str(val),print_log=False)
    # ]
# ) 

# mean_post_transforms1 = Compose(
    # [
        # EnsureTyped(keys=["pred"+str(i) for i in range(len(models1))]), #gives pred0..pred1...
        # SplitChanneld(keys=["pred"+str(i) for i in range(10)]),
       
                        
        # MeanEnsembled(
            # keys=["pred"+str(i) for i in range(len(models1))], 
            # output_key="pred",
          #  # in this particular example, we use validation metrics as weights
          ## weights=wts,
        # ),
        # Activationsd(keys=["pred"+str(i) for i in range(len(models1))], sigmoid=True),
        # AsDiscreted(keys=["pred"+str(i) for i in range(len(models1))], threshold=0.5),
    # ]
# )

# post_trans = Compose(
    # [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
# )
device=torch.device("cuda:0")
##invertd is only necessary if val_transform only works on image or when submitting to external dataset

# train_transform_isles = Compose(
    # [
        # LoadImaged(keys=["image", "mask"]),
        # EnsureChannelFirstD(keys=["image","mask"]),
        # # AddChannelD(keys="mask"), 
        # SpacingD(
            # keys=["image","mask"],
            # pixdim=(1.0, 1.0, 1.0),
            # mode="bilinear",
        # ),
        # OrientationD(keys=["image","mask"],axcodes="RAS"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  
        # # RandSpatialCropd(
        # # ["image"], roi_size=roi, random_size=False
        # # ),
        # CenterSpatialCropd(keys=["image","mask"], roi_size=roi),
        # EnsureTyped(keys=["image", "mask"]),
    # ]
# )
val_transform_isles = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        CastToTyped(keys=['image'],dtype=np.float32),
        EnsureChannelFirstD(keys=["image","mask"]),
        EnsureTyped(keys=["image", "mask"]),
        # AddChannelD(keys="mask"), 
        SpacingD(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
        ),
        # OrientationD(keys=["image"],axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  
        # RandSpatialCropd(
        # ["image"], roi_size=roi, random_size=False
        # ),
        # CenterSpatialCropd(keys=["image","mask"], roi_size=roi),
        
    ]
)
    
# post_trans = Compose(
    # [        
        # Invertd(
            # keys=["pred"],
            # transform=val_transform_isles,
            # orig_keys="image",
            # meta_keys=["pred_meta_dict"],
            # orig_meta_keys="image_meta_dict",
            # meta_key_postfix="meta_dict",
            # nearest_interp=False,
            # to_tensor=True,
            # device="cuda",
        # ), 
        # Activationsd(keys="pred", sigmoid=True),
        # AsDiscreted(keys="pred", threshold=0.5),
        
        # # KeepLargestConnectedComponentd(keys="pred",applied_labels=[0,1,2],independent=True,connectivity=3,num_components=2),
        # # RemoveSmallObjectsd(keys="pred",min_size=64, connectivity=1, independent_channels=True, by_measure=False, pixdim=None)
        # # ConvertToSingleChannel(keys='pred'),
        # # ConvertToMultiChannelBasedOnBratsClassesd_val(keys="pred"),
    # ]
# )


post_trans = Compose(
    [        
        Invertd(
            keys=["pred"],
            transform=val_transform_isles,
            orig_keys="image",
            meta_keys=["pred_meta_dict"],
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cuda",
        ), 
        Activationsd(keys="pred", softmax=True, dim=0),  # Apply softmax along channel dimension
        AsDiscreted(
            keys="pred",
            argmax=True,  # Use argmax to get the class with highest probability
            dim=0,
            
        ),
    ]
)
          

test_transforms0 = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstD(keys=["image"]),
        ConvertToMultiChannelBasedOnBratsClassesd_val(keys="mask"),
        
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # ToTensorD(keys=["image"], dtype=torch.float),
        EnsureTyped(keys=["image", "mask"]),
    ]
    )
test_transforms1 = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstD(keys=["image"]), 
        SpacingD(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        OrientationD(keys=["image"], axcodes="RAS"),
        
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # ToTensorD(keys=["image"], dtype=torch.float),
        EnsureTyped(keys=["image"]),
    ]
    )
    
transformer_transform = Compose(
[
    LoadImaged(keys=["image"]),
    EnsureChannelFirstD(keys=["image"]), 
    SpacingD(
        keys=["image"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear"),
    ),
    CenterSpatialCropd(keys=["image"], roi_size=roi),
    OrientationD(keys=["image"], axcodes="RAS"),
    
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    # ToTensorD(keys=["image"], dtype=torch.float),
    EnsureTyped(keys=["image"]),
]
)
    
post_trans_test = Compose(
    [           
        
        
        Invertd(
            keys="pred",
            transform=test_transforms1,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
        ),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
        KeepLargestConnectedComponentd(keys="pred",applied_labels=[0,1,2],independent=True,connectivity=3,num_components=1),
        RemoveSmallObjectsd(keys="pred",min_size=100, connectivity=3, independent_channels=True, by_measure=False, pixdim=None),
        ConvertToSingleChannel(keys='pred'),
        
        
    ]
)
probability=base_transform_probability
isles_list = [
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstD(keys=["image", "mask"]),
    CropForegroundd(keys=["image", "mask"], source_key="image"),
    # CastToTyped(keys=['image'], dtype=np.float32),
    SpacingD(
        keys=["image", "mask"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "nearest"),
    ),
    OrientationD(keys=["image", "mask"], axcodes="RAS"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    # CenterSpatialCropd(keys=["image","mask"], roi_size=roi),
    RandSpatialCropd(
        keys=["image", "mask"],
        roi_size=roi,
        random_size=False
    ),
    # Padding to ensure image is at least the roi_size
    SpatialPadd(keys=["image", "mask"], spatial_size=roi),

    EnsureTyped(keys=["image", "mask"]),
    
    # RandRotateD(
        # keys=["image", "mask"],
        # range_x=np.pi/12,
        # range_y=np.pi/12,
        # range_z=np.pi/12,
        # prob=probability,
        # mode=("bilinear", "nearest"),
        # padding_mode="border",
    # ),
    RandAffined(
        keys=["image", "mask"],
        prob=probability,
        rotate_range=(np.pi/12, np.pi/12, np.pi/12),
        scale_range=(0.2, 0.2, 0.2),
        spatial_size=roi,
        cache_grid=True,
        mode=("bilinear", "nearest"),
        padding_mode="border",
    ),

    
    
    RandGaussianSmoothd(
        keys="image",
        prob=probability,
        sigma_x=(0.5, 1),
        sigma_y=(0.5, 1),
        sigma_z=(0.5, 1),
    ),
    
    RandScaleIntensityd(keys="image", factors=0.3, prob=min(probability+0.3,1)),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=min(probability+0.3,1)),
    RandGaussianNoised(keys="image", prob=probability, mean=0.0, std=0.1),
    RandFlipd(keys=["image", "mask"], spatial_axis=[0, 1, 2], prob=min(probability+0.3,1)),
]

train_transform_isles = Compose(
    isles_list
)
class DynamicProbabilityTransform:
    def __init__(self, transform, start_prob=0.0):
        self.transform = transform
        self.start_prob = start_prob
        self.current_prob = start_prob

    def set_probability(self, init_loss, best_loss,patience):
        # loss scaling of probability
        self.current_prob = self.start_prob + (1-self.start_prob)*((init_loss - best_loss)/(patience*init_loss))
        print('augmentation shared probability set to ', self.current_prob)
    def __call__(self, x):  
        rn= random.random()
        if rn < self.current_prob:
            return self.transform(x)
        return x

# Ad-hoc transform update in the training loop
def update_transforms_for_epoch(x_transform, init_loss, best_loss,patience=2):
    transform_list = []
    
    for i, transform in enumerate(x_transform):
        if i<(len(isles_list)-6):
             transform_list.append(transform)
        else:
            dynamic_transform = DynamicProbabilityTransform(transform, start_prob=0.0)
            dynamic_transform.set_probability(init_loss, best_loss,patience)  # Adjust probability based on epoch
            transform_list.append(dynamic_transform)
    
    # Rebuild the compose transform pipeline with updated probabilities
    x_transform = Compose(transform_list)

    return x_transform
    

# List modification function
def factor_increment(init_loss,best_loss,base_probability=0.3):
    factor=3*(init_loss-best_loss)/init_loss
    isles_list = [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstD(keys=["image", "mask"]),
        CastToTyped(keys=['image'], dtype=np.float32),
        SpacingD(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        OrientationD(keys=["image", "mask"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandSpatialCropd(
            keys=["image", "mask"],
            roi_size=roi,
            random_size=False
        ),
        SpatialPadd(keys=["image", "mask"], spatial_size=roi),
        EnsureTyped(keys=["image", "mask"]),

        # Augmentations with modified factors
        RandRotateD(
            keys=["image", "mask"],
            range_x=np.pi / 12 * factor,
            range_y=np.pi / 12 * factor,
            range_z=np.pi / 12 * factor,
            prob=base_probability,
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        RandAffined(
            keys=["image", "mask"],
            prob=base_probability,
            rotate_range=(np.pi / 12 * factor, np.pi / 12 * factor, np.pi / 12 * factor),
            scale_range=(0.1 * factor, 0.1 * factor, 0.1 * factor),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        RandFlipd(keys=["image", "mask"], spatial_axis=[0, 1, 2], prob=base_probability),
        RandGaussianSmoothd(
            keys="image",
            prob=base_probability,
            sigma_x=(0.8 * factor, 1.2 * factor),
            sigma_y=(0.8 * factor, 1.2 * factor),
            sigma_z=(0.8 * factor, 1.2 * factor),
        ),
        RandGaussianNoised(keys="image", prob=base_probability, mean=0.0, std=0.1 * factor),
        RandScaleIntensityd(keys="image", factors=0.3 * factor, prob=base_probability),
        RandShiftIntensityd(keys="image", offsets=0.1 * factor, prob=base_probability),
    ]

    return isles_list