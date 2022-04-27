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



from monai.losses import DiceLoss,GeneralizedDiceLoss
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
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction, Weight, look_up_option

class SizeDiceLoss(_Loss):
    """
    Compute the generalised Dice loss defined in:

        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279
    """

    def __init__(
        self,
        include_background = True,
        to_onehot_y= False,
        sigmoid = False,
        softmax= False,
        other_act = None,
        w_type = Weight.SQUARE,
        reduction = LossReduction.MEAN,
        smooth_nr = 1e-5,
        smooth_dr = 1e-5,
        batch = False,
    ) -> None:
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            w_type: {``"square"``, ``"simple"``, ``"uniform"``}
                Type of function to transform ground truth volume to a weight factor. Defaults to ``"square"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, intersection over union is computed from each item in the batch.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act

        self.w_type = look_up_option(w_type, Weight)

        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def w_func(self, grnd):
        if self.w_type == Weight.SIMPLE:
            return torch.reciprocal(grnd)
        if self.w_type == Weight.SQUARE:
            return torch.reciprocal(grnd * grnd)
        return torch.ones_like(grnd)

    def forward(self, input, target):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis
        intersection = torch.sum(target * input, reduce_axis)

        ground_o = torch.sum(target, reduce_axis)
        pred_o = torch.sum(input, reduce_axis)
        factor=1+torch.square(ground_o-pred_o)/(torch.square(ground_o)+torch.square(pred_o)+1)
        
        
        denominator = ground_o + pred_o

        w = self.w_func(ground_o.float())
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)

        final_reduce_dim = 0 if self.batch else 1
        numer = 2.0 * (intersection * w).sum(final_reduce_dim, keepdim=True) + self.smooth_nr
        denom = (denominator * w).sum(final_reduce_dim, keepdim=True) + self.smooth_dr
        f = (1.0 - (numer / denom))*factor

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f

class CompleteLoss(SizeDiceLoss):
    def __init__(
        self,
        include_background = True,
        to_onehot_y= False,
        sigmoid = False,
        softmax= False,
        other_act = None,
        w_type = Weight.SQUARE,
        reduction = LossReduction.MEAN,
        smooth_nr = 1e-5,
        smooth_dr = 1e-5,
        batch = False
    ):
    super().__init__()
    self.include_background = include_background
    self.to_onehot_y = to_onehot_y
    self.sigmoid = sigmoid
    self.softmax = softmax
    self.other_act = other_act

    self.w_type = look_up_option(w_type, Weight)

    self.smooth_nr = float(smooth_nr)
    self.smooth_dr = float(smooth_dr)
    self.batch = batch
    
    def forward(self, input, target):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis
        intersection = torch.sum(target * input, reduce_axis)
        
        target_ind=torch.nonzero(target)
        target_mean=torch.sum(target_ind)/len(target_ind)
        
        input_ind=torch.nonzero(input)
        input_mean=torch.sum(input_ind)/len(input_ind)
        
        
        ground_o = torch.sum(target, reduce_axis)
        pred_o = torch.sum(input, reduce_axis)
        
        factor=1+torch.square(ground_o-pred_o)/(torch.square(ground_o)+torch.square(pred_o)+1)
        dist_factor=1+torch.square(input_mean-target_mean)/(torch.square(target_mean)+torch.square(input_mean)+1)
        
        
        denominator = ground_o + pred_o

        w = self.w_func(ground_o.float())
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)

        final_reduce_dim = 0 if self.batch else 1
        numer = 2.0 * (intersection * w).sum(final_reduce_dim, keepdim=True) + self.smooth_nr
        denom = (denominator * w).sum(final_reduce_dim, keepdim=True) + self.smooth_dr
        f = (1.0 - (numer / denom))*factor*dist_factor

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f

        
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
    parser.add_argument("--seed",default=0,type=int, help="random seed for the script")
    parser.add_argument("--method",default='A', type=str,help='A,B or C')
    

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
                train_indices=np.delete(indexes,val_indices)#indexes[(i-5)*fold:(i-4)*fold]
                
               
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

    #dataset=DecathlonDataset(root_dir="./", task="Task05_Prostate",section="training", transform=xform, download=True)
    train_dataset=BratsDataset("./RSNA_ASNR_MICCAI_BraTS2021_TrainingData"  ,transform=train_transform ) 

    


    if args.CV_flag==1:
        print("loading cross val data")
        val_dataset=Subset(train_dataset,val_indices)
        train_dataset=Subset(train_dataset,train_indices)
        
    # else:     
        # print("loading data for single model training")
        # val_dataset=Subset(train_dataset,np.arange(800,1000))
        # train_dataset=Subset(train_dataset,np.arange(800))
        
        
    print("number of files processed: ", train_dataset.__len__())
    train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader=DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print("All Datasets assigned")

    root_dir="./"

    max_epochs = args.epochs
    val_interval = 1
    VAL_AMP = True

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

    else:
        model = locals() [args.model](4,3).to(device)

    with torch.cuda.amp.autocast():
        summary(model,(4,192,192,144))

    model=torch.nn.DataParallel(model)
    print("Model defined and passed to GPU")

    loss_function = locals()[args.method](smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True)
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
        # lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        # if (epoch+1)>80:
            # torch.save(
                            # model.state_dict(),
                            # os.path.join(root_dir, args.model+"ep"+str(epoch+1)+"rs"+str(args.seed)+args.method))

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
                            os.path.join(root_dir, date.today().isoformat()+ args.model+"CV"+str(args.fold_num)+"ms"+str(args.max_samples)),
                        )
                    else:
                        torch.save(
                            model.state_dict(),
                            os.path.join(root_dir, args.model+"ep"+str(epoch+1)+"rs"+str(args.seed)+args.method)
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

