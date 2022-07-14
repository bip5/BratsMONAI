from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
from monai.engines import SupervisedTrainer
from torch.nn.parallel import DistributedDataParallel
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

torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'

task_name = {
    "001": "Task001_BrainTumour",
    "500": "Task500_BraTS"
   
}

clip_values = {
    "001": [0, 0],
    "500": [0, 0]
    
}

normalize_values = {
    "001": [0, 0],
    "500": [0, 0],
    
}

deep_supr_num = {
    "001": 3,
    "500": 3,
  
}

patch_size = {
    "001": [128, 128, 128],
    "500": [128, 128, 128],
 
}

spacing = { 
    "001": [1.0, 1.0, 1.0],
    "500": [1.0, 1.0, 1.0],
    }
    ## makes a list of all image paths inside a directory
def make_dataset(data_dir):
        all_files = []
        images=[]
        masks=[]
        im_temp=[]
        datalist=[]
        assert os.path.isdir(data_dir), '%s is not a valid directory' % data_dir
        
        for root, fol, _ in sorted(os.walk(data_dir)): # list folders and root
            for folder in fol:                    # for each folder
                 path=os.path.join(root, folder)  # combine root path with folder path
                 for root1, _, fnames in os.walk(path):       #list all file names in the folder         
                    for f in fnames:                          # go through each file name
                        fpath=os.path.join(root1,f)
                        if is_image_file(f):                  # check if expected extension
                            if re.search("seg",f):            # look for the mask files- have'seg' in the name 
                                masks.append( fpath)
                            else:
                                im_temp.append(fpath)         # all without seg are image files, store them in a list for each folder
                    images.append(im_temp)                    # add image files for each folder to a list
                    im_temp=[]
                    
        for image,label in zip(images,masks):
            datalist.append({'image':image,'label':label})
        return datalist

def get_task_transforms(mode, task_id, pos_sample_num, neg_sample_num, num_samples):
    if mode != "test":
        keys = ["image", "label"]
    else:
        keys = ["image"]

    load_transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]
    # 2. sampling
    sample_transforms = [
        PreprocessAnisotropic(
            keys=keys,
            clip_values=clip_values[task_id],
            pixdim=spacing[task_id],
            normalize_values=normalize_values[task_id],
            model_mode=mode,
        ),
    ]
    # 3. spatial transforms
    if mode == "train":
        other_transforms = [
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size[task_id]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size[task_id],
                pos=pos_sample_num,
                neg=neg_sample_num,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.9,
                max_zoom=1.2,
                mode=("trilinear", "nearest"),
                align_corners=(True, None),
                prob=0.15,
            ),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.15,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
            RandFlipd(["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=[2], prob=0.5),
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
        ]
    elif mode == "validation":
        other_transforms = [
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
        ]
    else:
        other_transforms = [
            CastToTyped(keys=["image"], dtype=(np.float32)),
            EnsureTyped(keys=["image"]),
        ]

    all_transforms = load_transforms + sample_transforms + other_transforms
    return Compose(all_transforms)


def resample_image(image, shape, anisotrophy_flag):
    resized_channels = []
    if anisotrophy_flag:
        for image_c in image:
            resized_slices = []
            for i in range(image_c.shape[-1]):
                image_c_2d_slice = image_c[:, :, i]
                image_c_2d_slice = resize(
                    image_c_2d_slice,
                    shape[:-1],
                    order=3,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                resized_slices.append(image_c_2d_slice)
            resized = np.stack(resized_slices, axis=-1)
            resized = resize(
                resized,
                shape,
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    else:
        for image_c in image:
            resized = resize(
                image_c,
                shape,
                order=3,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    resized = np.stack(resized_channels, axis=0)
    return resized


def resample_label(label, shape, anisotrophy_flag):
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = np.max(label)
    if anisotrophy_flag:
        shape_2d = shape[:-1]
        depth = label.shape[-1]
        reshaped_2d = np.zeros((*shape_2d, depth), dtype=np.uint8)

        for class_ in range(1, int(n_class) + 1):
            for depth_ in range(depth):
                mask = label[0, :, :, depth_] == class_
                resized_2d = resize(
                    mask.astype(float),
                    shape_2d,
                    order=1,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped_2d[:, :, depth_][resized_2d >= 0.5] = class_
        for class_ in range(1, int(n_class) + 1):
            mask = reshaped_2d == class_
            resized = resize(
                mask.astype(float),
                shape,
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_
    else:
        for class_ in range(1, int(n_class) + 1):
            mask = label[0] == class_
            resized = resize(
                mask.astype(float),
                shape,
                order=1,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_

    reshaped = np.expand_dims(reshaped, 0)
    return reshaped


def recovery_prediction(prediction, shape, anisotrophy_flag):
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = shape[0]
    if anisotrophy_flag:
        c, h, w = prediction.shape[:-1]
        d = shape[-1]
        reshaped_d = np.zeros((c, h, w, d), dtype=np.uint8)
        for class_ in range(1, n_class):
            mask = prediction[class_] == 1
            resized_d = resize(
                mask.astype(float),
                (h, w, d),
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped_d[class_][resized_d >= 0.5] = 1

        for class_ in range(1, n_class):
            for depth_ in range(d):
                mask = reshaped_d[class_, :, :, depth_] == 1
                resized_hw = resize(
                    mask.astype(float),
                    shape[1:-1],
                    order=1,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped[class_, :, :, depth_][resized_hw >= 0.5] = 1
    else:
        for class_ in range(1, n_class):
            mask = prediction[class_] == 1
            resized = resize(
                mask.astype(float),
                shape[1:],
                order=1,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[class_][resized >= 0.5] = 1

    return reshaped


class PreprocessAnisotropic(MapTransform):
    """
        This transform class takes NNUNet's preprocessing method for reference.
        That code is in:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py

    """

    def __init__(
        self,
        keys,
        clip_values,
        pixdim,
        normalize_values,
        model_mode,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.low = clip_values[0]
        self.high = clip_values[1]
        self.target_spacing = pixdim
        self.mean = normalize_values[0]
        self.std = normalize_values[1]
        self.training = False
        self.crop_foreg = CropForegroundd(keys=["image", "label"], source_key="image")
        self.normalize_intensity = NormalizeIntensity(nonzero=True, channel_wise=True)
        if model_mode in ["train"]:
            self.training = True

    def calculate_new_shape(self, spacing, shape):
        spacing_ratio = np.array(spacing) / np.array(self.target_spacing)
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def check_anisotrophy(self, spacing):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def __call__(self, data):
        # load data
        d = dict(data)
        image = d["image"]
        image_spacings = d["image_meta_dict"]["pixdim"][1:4].tolist()

        if "label" in self.keys:
            label = d["label"]
            label[label < 0] = 0

        if self.training:
            # only task 04 does not be impacted
            cropped_data = self.crop_foreg({"image": image, "label": label})
            image, label = cropped_data["image"], cropped_data["label"]
        else:
            d["original_shape"] = np.array(image.shape[1:])
            box_start, box_end = generate_spatial_bounding_box(image)
            image = SpatialCrop(roi_start=box_start, roi_end=box_end)(image)
            d["bbox"] = np.vstack([box_start, box_end])
            d["crop_shape"] = np.array(image.shape[1:])

        original_shape = image.shape[1:]
        # calculate shape
        resample_flag = False
        anisotrophy_flag = False
        if self.target_spacing != image_spacings:
            # resample
            resample_flag = True
            resample_shape = self.calculate_new_shape(image_spacings, original_shape)
            anisotrophy_flag = self.check_anisotrophy(image_spacings)
            image = resample_image(image, resample_shape, anisotrophy_flag)
            if self.training:
                label = resample_label(label, resample_shape, anisotrophy_flag)

        d["resample_flag"] = resample_flag
        d["anisotrophy_flag"] = anisotrophy_flag
        # clip image for CT dataset
        if self.low != 0 or self.high != 0:
            image = np.clip(image, self.low, self.high)
            image = (image - self.mean) / self.std
        else:
            image = self.normalize_intensity(image.copy())

        d["image"] = image

        if "label" in self.keys:
            d["label"] = label

        return d
        
def get_data(args,batch_size=1, mode="train"):
    # get necessary parameters:
    fold = args.fold
    task_id = args.task_id
    root_dir = args.root_dir
    datalist_path = args.datalist_path
    dataset_path = os.path.join(root_dir, task_name[task_id])
    transform_params = (args.pos_sample_num, args.neg_sample_num, args.num_samples)
    multi_gpu_flag = args.multi_gpu

    transform = get_task_transforms(mode, task_id, *transform_params)
    if mode == "test":
        list_key = "test"
    else:
        list_key = "{}_fold{}".format(mode, fold)
    datalist_name = "dataset_task{}.json".format(task_id)

    property_keys = [
        "name",
        "description",
        "reference",
        "licence",
        "tensorImageSize",
        "modality",
        "labels",
        "numTraining",
        "numTest",
    ]

    datalist = load_decathlon_datalist(
        os.path.join(datalist_path, datalist_name), True, list_key, dataset_path
    )

    properties = load_decathlon_properties(
        os.path.join(datalist_path, datalist_name), property_keys)
    if mode in ["validation", "test"]:
        if multi_gpu_flag:
            datalist = partition_dataset(
                data=datalist,
                shuffle=True,
                num_partitions=dist.get_world_size(),
                even_divisible=True,
            )[dist.get_rank()]

        val_ds = CacheDataset(
            data=datalist,
            transform=transform,
            num_workers=args.val_num_workers,
        )

        data_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.val_num_workers,
        )
    elif mode == "train":
        if multi_gpu_flag:
            datalist = partition_dataset(
                data=datalist,
                shuffle=True,
                num_partitions=dist.get_world_size(),
                even_divisible=True,
            )[dist.get_rank()]

        train_ds = CacheDataset(
            data=datalist,
            transform=transform,
            num_workers=args.train_num_workers,
            cache_rate=args.cache_rate,
            
        )
        data_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.train_num_workers,
            drop_last=True,
            
        )
    else:
        raise ValueError(f"mode should be train, validation or test.")

    return properties, data_loader
    
def get_kernels_strides(task_id):
    sizes, spacings = patch_size[task_id], spacing[task_id]
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
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


def get_network(properties, task_id, pretrain_path, checkpoint=None):
    n_class = len(properties["labels"])
    in_channels = len(properties["modality"])
    kernels, strides = get_kernels_strides(task_id)

    net = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_class,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=deep_supr_num[task_id],
    )
    
    if checkpoint is not None:
        pretrain_path = os.path.join(pretrain_path, checkpoint)
        if os.path.exists(pretrain_path):
            net.load_state_dict(torch.load(pretrain_path))
            print("pretrained checkpoint: {} loaded".format(pretrain_path))
        else:
            print("no pretrained checkpoint")
    return net
    
class DynUNetEvaluator(SupervisedEvaluator):


    def __init__(
        self,
        device: torch.device,
        val_data_loader: DataLoader,
        network: torch.nn.Module,
        num_classes: Union[str, int],
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        postprocessing: Optional[Transform] = None,
        key_val_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        val_handlers: Optional[Sequence] = None,
        amp: bool = False,
        tta_val: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            network=network,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            inferer=inferer,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            val_handlers=val_handlers,
            amp=amp,
        )

        if not isinstance(num_classes, int):
            num_classes = int(num_classes)
        self.num_classes = num_classes
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label = AsDiscrete(to_onehot=num_classes)
        self.tta_val = tta_val

    def _iteration(
        self, engine: Engine, batchdata: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:

        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, targets = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, targets, args, kwargs = batch

        targets = targets.cpu()

        def _compute_pred():
            ct = 1.0
            pred = self.inferer(inputs, self.network, *args, **kwargs).cpu()
            pred = nn.functional.softmax(pred, dim=1)
            if not self.tta_val:
                return pred
            else:
                for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                    flip_inputs = torch.flip(inputs, dims=dims)
                    flip_pred = torch.flip(
                        self.inferer(flip_inputs, self.network).cpu(), dims=dims
                    )
                    flip_pred = nn.functional.softmax(flip_pred, dim=1)
                    del flip_inputs
                    pred += flip_pred
                    del flip_pred
                    ct += 1
                return pred / ct

        # execute forward computation
        with eval_mode(self.network):
            if self.amp:
                with torch.cuda.amp.autocast():
                    predictions = _compute_pred()
            else:
                predictions = _compute_pred()

        inputs = inputs.cpu()

        predictions = self.post_pred(decollate_batch(predictions)[0])
        targets = self.post_label(decollate_batch(targets)[0])

        resample_flag = batchdata["resample_flag"]
        anisotrophy_flag = batchdata["anisotrophy_flag"]
        crop_shape = batchdata["crop_shape"][0].tolist()
        original_shape = batchdata["original_shape"][0].tolist()
        if resample_flag:
            # convert the prediction back to the original (after cropped) shape
            predictions = recovery_prediction(
                predictions.numpy(), [self.num_classes, *crop_shape], anisotrophy_flag
            )
            predictions = torch.tensor(predictions)

        # put iteration outputs into engine.state
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets.unsqueeze(0)}
        engine.state.output[Keys.PRED] = torch.zeros([1, self.num_classes, *original_shape])
        # pad the prediction back to the original shape
        box_start, box_end = batchdata["bbox"][0]
        h_start, w_start, d_start = box_start
        h_end, w_end, d_end = box_end

        engine.state.output[Keys.PRED][
            0, :, h_start:h_end, w_start:w_end, d_start:d_end
        ] = predictions
        del predictions

        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output
        
        
class DynUNetTrainer(SupervisedTrainer):
    """
    This class inherits from SupervisedTrainer in MONAI, and is used with DynUNet
    on Decathlon datasets.

    """

    def _iteration(self, engine: Engine, batchdata: Dict[str, Any]):
        """
        Callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.
            - LOSS: loss value computed by loss function.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, targets = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, targets, args, kwargs = batch
        # put iteration outputs into engine.state
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets}

        def _compute_pred_loss():
            preds = self.inferer(inputs, self.network, *args, **kwargs)
            if len(preds.size()) - len(targets.size()) == 1:
                # deep supervision mode, need to unbind feature maps first.
                preds = torch.unbind(preds, dim=1)
            engine.state.output[Keys.PRED] = preds
            del preds
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            engine.state.output[Keys.LOSS] = sum(
                0.5 ** i * self.loss_function.forward(p, targets)
                for i, p in enumerate(engine.state.output[Keys.PRED])
            )
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

        self.network.train()
        self.optimizer.zero_grad()
        if self.amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                _compute_pred_loss()
            self.scaler.scale(engine.state.output[Keys.LOSS]).backward()
            self.scaler.unscale_(self.optimizer)
            if isinstance(self.network, DistributedDataParallel):
                torch.nn.utils.clip_grad_norm_(self.network.module.parameters(), 12)
            else:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            _compute_pred_loss()
            engine.state.output[Keys.LOSS].backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            if isinstance(self.network, DistributedDataParallel):
                torch.nn.utils.clip_grad_norm_(self.network.module.parameters(), 12)
            else:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output

def validation(args):
    # load hyper parameters
    task_id = args.task_id
    sw_batch_size = args.sw_batch_size
    tta_val = args.tta_val
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap
    multi_gpu_flag = args.multi_gpu
    local_rank = args.local_rank
    amp = args.amp

    # produce the network
    checkpoint = args.checkpoint
    val_output_dir = "./runs_{}_fold{}_{}/".format(task_id, args.fold, args.expr_name)

    if multi_gpu_flag:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda")

    properties, val_loader = get_data(args, mode="validation")
    net = get_network(properties, task_id, val_output_dir, checkpoint)
    net = net.to(device)

    if multi_gpu_flag:
        net = DistributedDataParallel(module=net, device_ids=[device])

    num_classes = len(properties["labels"])

    net.eval()

    evaluator = DynUNetEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        num_classes=num_classes,
        inferer=SlidingWindowInferer(
            roi_size=patch_size[task_id],
            sw_batch_size=sw_batch_size,
            overlap=eval_overlap,
            mode=window_mode,
        ),
        postprocessing=None,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=False,
                output_transform=from_engine(["pred", "label"]),
            )
        },
        additional_metrics=None,
        amp=amp,
        tta_val=tta_val,
    )

    evaluator.run()
    if local_rank == 0:
        print(evaluator.state.metrics)
        results = evaluator.state.metric_details["val_mean_dice"]
        if num_classes > 2:
            for i in range(num_classes - 1):
                print(
                    "mean dice for label {} is {}".format(i + 1, results[:, i].mean())
                )



def train(args):
    # load hyper parameters
    task_id = args.task_id
    fold = args.fold
    val_output_dir = "./runs_{}_fold{}_{}/".format(task_id, fold, args.expr_name)
    log_filename = "nnunet_task{}_fold{}.log".format(task_id, fold)
    log_filename = os.path.join(val_output_dir, log_filename)
    interval = args.interval
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    multi_gpu_flag = args.multi_gpu
    amp_flag = args.amp
    lr_decay_flag = args.lr_decay
    sw_batch_size = args.sw_batch_size
    tta_val = args.tta_val
    batch_dice = args.batch_dice
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap
    local_rank = args.local_rank
    determinism_flag = args.determinism_flag
    determinism_seed = args.determinism_seed
    if determinism_flag:
        set_determinism(seed=determinism_seed)
        if local_rank == 0:
            print("Using deterministic training.")

    # transforms
    train_batch_size = data_loader_params[task_id]["batch_size"]
    if multi_gpu_flag:
        dist.init_process_group(backend="nccl", init_method="env://")

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda")

    properties, val_loader = get_data(args, mode="validation")
    _, train_loader = get_data(args, batch_size=train_batch_size, mode="train")

    # produce the network
    checkpoint = args.checkpoint
    net = get_network(properties, task_id, val_output_dir, checkpoint)
    net = net.to(device)

    if multi_gpu_flag:
        net = DistributedDataParallel(module=net, device_ids=[device])

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9
    )
    # produce evaluator
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        CheckpointSaver(
            save_dir=val_output_dir, save_dict={"net": net}, save_key_metric=True
        ),
    ]

    evaluator = DynUNetEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        num_classes=len(properties["labels"]),
        inferer=SlidingWindowInferer(
            roi_size=patch_size[task_id],
            sw_batch_size=sw_batch_size,
            overlap=eval_overlap,
            mode=window_mode,
        ),
        postprocessing=None,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=False,
                output_transform=from_engine(["pred", "label"]),
            )
        },
        val_handlers=val_handlers,
        amp=amp_flag,
        tta_val=tta_val,
    )

    # produce trainer
    loss = DiceCELoss(to_onehot_y=True, softmax=True, batch=batch_dice)
    train_handlers = []
    if lr_decay_flag:
        train_handlers += [LrScheduleHandler(lr_scheduler=scheduler, print_lr=True)]

    train_handlers += [
        ValidationHandler(validator=evaluator, interval=interval, epoch_level=True),
        StatsHandler(
            tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
        ),
    ]

    trainer = DynUNetTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        postprocessing=None,
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp_flag,
    )

    if local_rank > 0:
        evaluator.logger.setLevel(logging.WARNING)
        trainer.logger.setLevel(logging.WARNING)

    logger = logging.getLogger()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Setup file handler
    fhandler = logging.FileHandler(log_filename)
    fhandler.setLevel(logging.INFO)
    fhandler.setFormatter(formatter)

    logger.addHandler(fhandler)

    chandler = logging.StreamHandler()
    chandler.setLevel(logging.INFO)
    chandler.setFormatter(formatter)
    logger.addHandler(chandler)

    logger.setLevel(logging.INFO)

    trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-fold", "--fold", type=int, default=0, help="0-5")
    parser.add_argument(
        "-task_id", "--task_id", type=str, default="500", help="task id"
    )
    parser.add_argument(
        "-root_dir",
        "--root_dir",
        type=str,
        default="./nnUNet_raw_data/",
        help="dataset path",
    )
    parser.add_argument(
        "-expr_name",
        "--expr_name",
        type=str,
        default="expr",
        help="the suffix of the experiment's folder",
    )
    parser.add_argument(
        "-datalist_path",
        "--datalist_path",
        type=str,
        default="config/",
    )
    parser.add_argument(
        "-train_num_workers",
        "--train_num_workers",
        type=int,
        default=4,
        help="the num_workers parameter of training dataloader.",
    )
    parser.add_argument(
        "-val_num_workers",
        "--val_num_workers",
        type=int,
        default=4,
        help="the num_workers parameter of validation dataloader.",
    )
    parser.add_argument(
        "-interval",
        "--interval",
        type=int,
        default=5,
        help="the validation interval under epoch level.",
    )
    parser.add_argument(
        "-eval_overlap",
        "--eval_overlap",
        type=float,
        default=0.5,
        help="the overlap parameter of SlidingWindowInferer.",
    )
    parser.add_argument(
        "-sw_batch_size",
        "--sw_batch_size",
        type=int,
        default=4,
        help="the sw_batch_size parameter of SlidingWindowInferer.",
    )
    parser.add_argument(
        "-window_mode",
        "--window_mode",
        type=str,
        default="gaussian",
        choices=["constant", "gaussian"],
        help="the mode parameter for SlidingWindowInferer.",
    )
    parser.add_argument(
        "-num_samples",
        "--num_samples",
        type=int,
        default=4,
        help="the num_samples parameter of RandCropByPosNegLabeld.",
    )
    parser.add_argument(
        "-pos_sample_num",
        "--pos_sample_num",
        type=int,
        default=3,
        help="the pos parameter of RandCropByPosNegLabeld.",
    )
    parser.add_argument(
        "-neg_sample_num",
        "--neg_sample_num",
        type=int,
        default=1,
        help="the neg parameter of RandCropByPosNegLabeld.",
    )
    parser.add_argument(
        "-cache_rate",
        "--cache_rate",
        type=float,
        default=1.0,
        help="the cache_rate parameter of CacheDataset.",
    )
    parser.add_argument("-learning_rate", "--learning_rate", type=float, default=1e-2)
    parser.add_argument(
        "-max_epochs",
        "--max_epochs",
        type=int,
        default=100,
        help="number of epochs of training.",
    )
    parser.add_argument(
        "-mode", "--mode", type=str, default="train", choices=["train", "val"]
    )
    parser.add_argument(
        "-checkpoint",
        "--checkpoint",
        type=str,
        default=None,
        help="the filename of weights.",
    )
    parser.add_argument(
        "-amp",
        "--amp",
        type=bool,
        default=False,
        help="whether to use automatic mixed precision.",
    )
    parser.add_argument(
        "-lr_decay",
        "--lr_decay",
        type=bool,
        default=False,
        help="whether to use learning rate decay.",
    )
    parser.add_argument(
        "-tta_val",
        "--tta_val",
        type=bool,
        default=False,
        help="whether to use test time augmentation.",
    )
    parser.add_argument(
        "-batch_dice",
        "--batch_dice",
        type=bool,
        default=False,
        help="the batch parameter of DiceCELoss.",
    )
    parser.add_argument(
        "-determinism_flag", "--determinism_flag", type=bool, default=False
    )
    parser.add_argument(
        "-determinism_seed",
        "--determinism_seed",
        type=int,
        default=0,
        help="the seed used in deterministic training",
    )
    parser.add_argument(
        "-multi_gpu",
        "--multi_gpu",
        type=bool,
        default=False,
        help="whether to use multiple GPUs for training.",
    )
    
    parser.add_argument(
        "-batch_size",
        "--batch_size",
        type=int,
        default=2,
        help="batch size.",
    )
    parser.add_argument("-local_rank", "--local_rank", type=int, default=0)
    
    args = parser.parse_args()
    print(' '.join(sys.argv))
    
    data_loader_params = {
    "001": {"batch_size": args.batch_size},
    "500": {"batch_size": args.batch_size}}
    if args.local_rank == 0:
        print_config()
    if args.mode == "train":
        train(args)
    elif args.mode == "val":
        validation(args)


        
