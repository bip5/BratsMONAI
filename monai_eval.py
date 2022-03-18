from typing import List, Optional, Sequence, Tuple, Union

import pandas
# print(pandas.__version__)
import nibabel

import os
import monai
from monai.data import Dataset
from monai.utils import set_determinism
from sisa import SISANet

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
    MeanEnsembled,
    VoteEnsembled,
    EnsureType,
    SplitChanneld,
)

from monai.engines import (
    EnsembleEvaluator,
    SupervisedEvaluator,
    SupervisedTrainer
)

from monai.losses import DiceLoss
from monai.utils import UpsampleMode
from monai.data import decollate_batch, list_data_collate
from monai.handlers import MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.networks.nets import SegResNet,UNet
from monai.metrics import DiceMetric,compute_meandice
from monai.inferers import sliding_window_inference
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.data import DataLoader
import numpy as np
from datetime import date, datetime
import sys
import re
import torch
import time
import argparse
from torchsummary import summary
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__=="__main__":
    parser=argparse.ArgumentParser(prog=sys.argv[0],description="Eval parser")

    parser.add_argument("--model",default="SISANet",type=str,help="name of model to use")
    parser.add_argument("--load_barlow",default =0, type=int,help="flag to use barlow twins backbone to initialise weight")
    parser.add_argument("--load_save",default =1, type=int,help="flag to use saved model weight")
    parser.add_argument("--load_name",default="./2022-01-20T16best_metric_model.pth", type=str, help="file path to load previously saved model")
    parser.add_argument("--batch_size",default=8, type=int, help="to define batch size")
    parser.add_argument("--save_name", default="Best_metric_model.pth",type=str, help="save name")
    parser.add_argument("--upsample", default="DECONV",type=str, help="flag to choose deconv options- NONTRAINABLE, DECONV, PIXELSHUFFLE")
    parser.add_argument("--barlow_final",default=1, type=int, help="flag to use checkpoint instead of final model for barlow")
    parser.add_argument("--bar_model_name",default="checkpoint.pth", type=str,help="model name to load")
    parser.add_argument("--max_samples",default=10000,type=int,help="max number of samples to use for training")
    parser.add_argument("--ensemble",default=0,type=int,help="flag to use ensemble with models provided")

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
        labels=[]
        im_temp=[]
        assert os.path.isdir(data_dir), '%s is not a valid directory' % data_dir
        
        for root, fol, _ in sorted(os.walk(data_dir)): # list folders and root
            for folder in fol:                    # for each folder
                 path=os.path.join(root, folder)  # combine root path with folder path
                 for root1, _, fnames in os.walk(path):       #list all file names in the folder         
                    for f in fnames:                          # go through each file name
                        fpath=os.path.join(root1,f)
                        if is_image_file(f):                  # check if expected extension
                            if re.search("seg",f):            # look for the label files- have'seg' in the name 
                                labels.append(fpath)
                            else:
                                im_temp.append(fpath)         # all without seg are image files, store them in a list for each folder
                    images.append(im_temp)                    # add image files for each folder to a list
                    im_temp=[]
        return images, labels

    # A source: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
    class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
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
                

    ##################~~~~~~~~~~~~~~~~~~~~~~~~~Model Definition~~~~~~~~~~~~~~~~~~~~~~~#################
    root_dir="./"

    max_epochs = 0
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



    ########### ~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~FIX this

    test_transforms0 = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstD(keys=["image"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            SpacingD(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            OrientationD(keys=["image", "label"], axcodes="RAS"),
            # RandSpatialCropd(keys=["image", "label"], roi_size=[192, 192, 144], random_size=False),
           
            RandRotateD(keys=["image","label"],range_x=0.1,range_y=0.1, range_z=0.1,prob=0.5),
           
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),

            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
        )


    class TestDataset(Dataset):
        def __init__(self,data_dir,transform=None):
            self.image_list=make_dataset(data_dir)[0]  
            self.label_list=make_dataset(data_dir)[1] 
            self.transform=transform
            
        def __len__(self):
    #         return len(os.listdir(self.label_dir))
            return min(args.max_samples,len(self.label_list))
        
        def __getitem__(self,idx):
            image=self.image_list[idx]
        
            label=self.label_list[idx] 

                
            item_dict={"image":image,"label":label}
            
            test_list=dict()
            if self.transform:
                item_dict={"image":image,"label": label}
                
                test_list=self.transform(item_dict)
                
            
            return test_list

    test_ds=TestDataset("./RSNA_ASNR_MICCAI_BraTS2021_TestData",transform=test_transforms0)



    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4) # this should return 10 different instances

    # print("input type",type(next(iter(test_loader))))





    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        # Invertd(
            # keys="pred",
            # transform=test_transforms0,
            # orig_keys="image",
            # meta_keys="pred_meta_dict",
            # orig_meta_keys="image_meta_dict",
            # meta_key_postfix="meta_dict",
            # nearest_interp=False,
            # to_tensor=True, #this sends to GPU so removing will cause problems
        # ), # inversal is only done on the prediction?
        ToTensorD(keys="pred"),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
    ])

    if args.ensemble==1:
    
        if args.model=="UNet":
            model_names=['2022-03-09T0UNetCV1','2022-03-09T1UNetCV2','2022-03-09T1UNetCV3','2022-03-09T1UNetCV4','2022-03-09T1UNetCV5','2022-03-09T1UNetCV6','2022-03-09T1UNetCV7','2022-03-09T1UNetCV8','2022-03-09T1UNetCV9','2022-03-09UNetCV10']
            wts=[0.69,0.69,0.78,0.72,0.62,0.7,0.7,0.7,0.75,0.67]
       
        elif args.model=="SegResNet":
            model_names=['2022-03-09T2SegResNetCV1','2022-03-09SegResNetCV2','2022-03-09SegResNetCV3','2022-03-09SegResNetCV4','2022-03-09SegResNetCV5','2022-03-09SegResNetCV6','2022-03-09SegResNetCV7','2022-03-09SegResNetCV8','2022-03-09SegResNetCV9','2022-03-09SegResNetCV10']
            wts=[0.8912,0.7912,0.6377,0.7312,0.7685,0.8118,0.6724,0.7227,0.6093,0.6378]
        

        else:
            print("No SISA yet")


        

        models=[]
        for i,name in enumerate(model_names):
            model.load_state_dict(torch.load("./saved models/"+name))
            model.eval()
            models.append(model)

        def ensemble_evaluate(post_transforms, models):
            evaluator = EnsembleEvaluator(
                device=device,
                val_data_loader=test_loader, #test dataloader - this is loading all 5 sets of data
                pred_keys=["pred"+str(i) for i in range(10)], 
                networks=models, # models defined above
                inferer=SlidingWindowInferer(
                    roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
                postprocessing=post_transforms, # this is going to call post_transforms based on type of ensemble
                key_val_metric={
                    "test_mean_dice": MeanDice(
                        include_background=True,
                        output_transform=from_engine(["pred", "label"])  # takes all the preds and labels and turns them into one list each
                        
                    )},
                additional_metrics={ 
                    "Channelwise": MeanDice(
                    include_background=True,
                    output_transform=from_engine(["pred", "label"]),
                    reduction="mean_batch")
                }
            )
            evaluator.run()
            
            # print("validation stats: ",evaluator.get_validation_stats())
            print("Mean Dice:",evaluator.state.metrics['test_mean_dice'],"metric_tc:",evaluator.state.metrics["Channelwise"][0],"whole tumor:",evaluator.state.metrics["Channelwise"][1],"enhancing tumor:",evaluator.state.metrics["Channelwise"][2])#jbc
            # print("evaluator best metric:",evaluator.state.best_metric)
        
        
        
        mean_post_transforms = Compose(
            [
                EnsureTyped(keys=["pred"+str(i) for i in range(10)]), #gives pred0..pred9
                # SplitChanneld(keys=["pred"+str(i) for i in range(10)]),
                
                MeanEnsembled(
                    keys=["pred"+str(i) for i in range(10)], 
                    output_key="pred",
                    # in this particular example, we use validation metrics as weights
                    weights=wts,
                ),
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold=0.5),
            ]
        )
        vote_post_transforms = Compose(
            [
                EnsureTyped(keys=["pred"+str(i) for i in range(10)]),
                Activationsd(keys=["pred"+str(i) for i in range(10)], sigmoid=True),
                # transform data into discrete before voting
                
                VoteEnsembled(keys=["pred"+str(i) for i in range(10)], output_key="pred"),
                AsDiscreted(keys="pred", threshold=0.5),
            ]
        )
        ensemble_evaluate(vote_post_transforms, models)
        # ensemble_evaluate(mean_post_transforms, models)

    else:
        
        model.load_state_dict(torch.load("./saved models/"+args.load_name))
        model.eval()

        with torch.no_grad():

            for test_data in test_loader: # each image
                test_inputs = test_data["image"].to(device) # pass to gpu
                test_data["pred"] = inference(test_inputs) #perform inference
                #print(test_data["pred"].shape)
                test_data=[post_transforms(i) for i in decollate_batch(test_data)] #reverse the transform and get sigmoid then binarise
                test_outputs, test_labels =  from_engine(["pred", "label"])(test_data) # create list of images and labels
              
                
                #print("test outputs",test_outputs[0].shape)
            
                test_labels[0]=test_labels[0].to(device)
                # dice_score=(2*torch.sum( test_outputs[0].flatten()*test_labels[0].flatten()))/( test_outputs[0].sum()+test_labels[0].sum())
                # print("dice",dice_score)
                
                dice_metric(y_pred=test_outputs, y=test_labels)
                dice_metric_batch(y_pred=test_outputs, y=test_labels)


            
            metric_org = dice_metric.aggregate().item()
            
            metric_batch_org = dice_metric_batch.aggregate()

            dice_metric.reset()
            dice_metric_batch.reset()

        metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()

        print("Metric on original image spacing: ", metric_org)
        print(f"metric_tc: {metric_tc:.4f}")
        print(f"metric_wt: {metric_wt:.4f}")
        print(f"metric_et: {metric_et:.4f}")


    #####################~~~~~~~~3D Seg End ~~~~~~###################


