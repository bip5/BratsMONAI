from monai.transforms import (EnsureChannelFirstD, ToMetaTensorD,\
    ToTensorD,
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
    CenterSpatialCropd,
    RandSpatialCropd,   
    EnsureTyped,
    EnsureType,
)
import torch
import numpy as np


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
            result.append(np.logical_or(d[key] == 3, d[key] == 1))
            # merge masks 1, 2 and 4 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # mask 4 is ET
            result.append(d[key] == 3)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d
            
KEYS=("image","mask")
print("Transforms not defined yet")
train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstD(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd_val(keys="mask"),
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
        ToTensorD(keys=["image"]),
        EnsureTyped(keys=["image", "mask"]),
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstD(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd_val(keys="mask"),
        SpacingD(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        OrientationD(keys=["image", "mask"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CenterSpatialCropd(keys=["image", "mask"], roi_size=[192, 192, 144]),
        # ToTensorD(keys=["image"], dtype=torch.float),
        EnsureTyped(keys=["image", "mask"]),
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

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
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
        
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # ToTensorD(keys=["image"], dtype=torch.float),
        EnsureTyped(keys=["image"]),
    ]
    )
