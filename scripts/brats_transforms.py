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
    GaussianSmoothD,
    RandGaussianNoised,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd, RandFlip,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,   
    ToTensorD,
    EnsureTyped,
    AdjustContrastD,
    RandKSpaceSpikeNoiseD,
    RandGaussianSharpenD,
    SaveImage,SaveImaged,

    MeanEnsembled,
    VoteEnsembled,
    EnsureType,
    Activations,
    SplitChanneld,
)

from eval_utils import *

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

test_transforms0 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstD(keys=["image"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ]
    )
