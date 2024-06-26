from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd
# print(pandas.__version__)
import nibabel

import os
import monai
from monai.data import Dataset
from monai.utils import set_determinism
from sisa import SISANet
import matplotlib.pyplot as plt

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

from monai.engines import (
    EnsembleEvaluator,
    SupervisedEvaluator,
    SupervisedTrainer
)
from monai.config.type_definitions import NdarrayOrTensor
from monai.networks import one_hot
from monai.networks.layers import GaussianFilter, apply_filter
from monai.transforms.transform import Transform
from monai.transforms.utils import fill_holes, get_largest_connected_component_mask
from monai.transforms.utils_pytorch_numpy_unification import unravel_index
from monai.utils import TransformBackends, convert_data_type, deprecated_arg, ensure_tuple, look_up_option
from monai.utils.type_conversion import convert_to_dst_type
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
from torch.utils.data import Subset
import argparse
from torchsummary import summary
import gc
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
    parser.add_argument("--avgmodel",default=0,type=int,help="flag to create an averaged model from existing models")
    parser.add_argument("--plot",default=0, type=int, help="plot=1,ensembleand plot, plot=2 evaluate all models in list")
    parser.add_argument("--val",default=0, type=int, help="val or not")
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
        label 2 is the peritumoral edema
        label 4 is the GD-enhancing tumor
        label 1 is the necrotic and non-enhancing tumor core
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor).

        """

        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                result = []
                # merge label 2 and label 3 to construct TC
                result.append(np.logical_or(d[key] == 1, d[key] == 4))
                # merge labels 1, 2 and 3 to construct WT
                result.append(
                    np.logical_or(
                        np.logical_or(d[key] == 2, d[key] == 4), d[key] == 1
                    )
                )
                # label 2 is ET
                result.append(d[key] == 4)
                d[key] = np.stack(result, axis=0).astype(np.float32)
            return d
                

    ##################~~~~~~~~~~~~~~~~~~~~~~~~~Model Definition~~~~~~~~~~~~~~~~~~~~~~~#################
    root_dir="./"

    max_epochs = 0
    val_interval = 1
    VAL_AMP = True
    saver_ori = SaveImage(output_dir='./ssensemblemodels0922/outputs', output_ext=".nii.gz", output_postfix="ori",print_log=True)
    saver_gt = SaveImage(output_dir='./ssensemblemodels0922/outputs', output_ext=".nii.gz", output_postfix="gt",print_log=True)
    saver_seg = SaveImage(output_dir='./ssensemblemodels0922/outputs', output_ext=".nii.gz", output_postfix="seg",print_log=True)

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
                overlap=0,
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
    
    
    class Ensemble:
        @staticmethod
        def get_stacked_torch(img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> torch.Tensor:
            """Get either a sequence or single instance of np.ndarray/torch.Tensor. Return single torch.Tensor."""
            if isinstance(img, Sequence) and isinstance(img[0], np.ndarray):
                print("converting pred list to tensor")
                img = [torch.as_tensor(i) for i in img]
            elif isinstance(img, np.ndarray):
                print("converting pred list to tensor")  
                img = torch.as_tensor(img)
            # else:
                # print("conversion wasn't necessary")
            
            out: torch.Tensor = torch.stack(img) if isinstance(img, Sequence) else torch.unsqueeze(img ,dim=0) # type: ignore
            return out

        @staticmethod
        def post_convert(img: torch.Tensor, orig_img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> NdarrayOrTensor:
            orig_img_ = orig_img[0] if isinstance(orig_img, Sequence) else orig_img
            out, *_ = convert_to_dst_type(img, orig_img_)
            return out
            

    class ConfEnsemble(Ensemble, Transform):


        backend = [TransformBackends.TORCH]

        def __init__(self, weights: Optional[Union[Sequence[float], NdarrayOrTensor]] = None) -> None:
            self.weights = torch.as_tensor(weights, dtype=torch.float) if weights is not None else None

        def __call__(self, img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> NdarrayOrTensor:
            img_ = self.get_stacked_torch(img)
            if self.weights is not None:
                self.weights = self.weights.to(img_.device)
                shape = tuple(self.weights.shape)
                for _ in range(img_.ndimension() - self.weights.ndimension()):
                    shape += (1,)
                weights = self.weights.reshape(*shape)

                img_ = img_ * weights / weights.mean(dim=0, keepdim=True)
                
            
            img_=torch.sort(img_,dim=0,descending=True).values #img is a tensor        
            
            # print(img_[0,2,100,100,70],img_[1,2,100,100,70],img_[2,2,100,100,70],img_[3,2,100,100,70],img_[4,2,100,100,70])
            out_pt = torch.mean(img_[0:10,:,:,:,:], dim=0)#[0:3,:,:,:,:]
            # print(out_pt.shape)
            x=self.post_convert(out_pt, img)
            
           
            return x
            
    class Ensembled(MapTransform):
       

        backend = list(set(ConfEnsemble.backend) )

        def __init__(
            self, keys,ensemble,
            output_key = None,
            allow_missing_keys = False,
        ) -> None:
       
            super().__init__(keys, allow_missing_keys)
            if not callable(ensemble):
                raise TypeError(f"ensemble must be callable but is {type(ensemble).__name__}.")
            self.ensemble = ensemble
            if len(self.keys) > 1 and output_key is None:
                raise ValueError("Incompatible values: len(self.keys) > 1 and output_key=None.")
            self.output_key = output_key if output_key is not None else self.keys[0]

        def __call__(self, data):
            d = dict(data)
            
            if len(self.keys) == 1 and self.keys[0] in d:
                items = d[self.keys[0]]
            else:
                items = [d[key] for key in self.key_iterator(d)]

            if len(items) > 0:
                d[self.output_key] = self.ensemble(items)

            return d
            
    class ConfEnsembled(Ensembled):
        """
        Dictionary-based wrapper of :py:class:`monai.transforms.MeanEnsemble`.
        """

        backend = ConfEnsemble.backend

        def __init__(
            self,
            keys,
            output_key = None,
            weights = None,
        ) -> None:
    
            ensemble = ConfEnsemble(weights=weights)
            super().__init__(keys, ensemble, output_key)

    test_transforms0 = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstD(keys=["image"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
        )

    indexes=np.arange(args.max_samples)
    fold=int(args.max_samples/10)
    
    class TestDataset(Dataset):
        def __init__(self,data_dir,transform=None):
            self.image_list=make_dataset(data_dir)[0]  
            self.label_list=make_dataset(data_dir)[1] 
            self.transform=transform
            
        def __len__(self):
    #         return len(os.listdir(self.label_dir))
            if args.val==1:
                return len(self.label_list)
            else:
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
            
    val_indices=[13,17]#np.arange(1000)
    

    
    if args.val==1:
        test_ds=TestDataset("./RSNA_ASNR_MICCAI_BraTS2021_TrainingData",transform=test_transforms0)
        test_ds=Subset(test_ds,val_indices)
    else:
        test_ds=TestDataset("./RSNA_ASNR_MICCAI_BraTS2021_TestData",transform=test_transforms0)
        # print("list of input image files",make_dataset("./RSNA_ASNR_MICCAI_BraTS2021_TestData")[0])



    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8) # this should return 10 different instances

    # print("input type",type(next(iter(test_loader))))





    post_transforms = Compose([
        EnsureTyped(keys=["pred","label"]), 
        Activationsd(keys="pred", sigmoid=True),
        Invertd(
            keys="pred",
            transform=test_transforms0,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            # meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True, #this sends to GPU so removing will cause problems
            device=device
        ), # inversal is only done on the prediction? yes with the specified key
        # ToTensorD(keys=["pred","label"]),
        
        AsDiscreted(keys="pred", threshold=0.5),
        SaveImaged(keys=["pred","label"],meta_keys="pred_meta_dict",output_dir="./ssensemblemodels0922/outputs/",resample=False)
    ])
    post_pred= Compose([EnsureType(),Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([EnsureType()])
    
    

    if args.ensemble==1:
    
        if args.model=="UNet":
            model_names= os.listdir('/scratch/a.bip5/BraTS 2021/ssensemblemodels0922/Evaluation Folder')
            #["UNetep100rs4C","UNetep99rs4C","UNetep98rs4C","UNetep97rs4C","UNetep96rs4C"]


            print(model_names)
            # wts=[0.69,0.69,0.78,0.72,0.62,0.7,0.7,0.7,0.75,0.67]
       
        elif args.model=="SegResNet":
            model_names=os.listdir('/scratch/a.bip5/BraTS 2021/ssensemblemodels0922/Evaluation Folder')#["SegResNetep100rs2C","SegResNetep99rs2C","SegResNetep98rs2C","SegResNetep97rs2C","SegResNetep96rs2C"]
            
           





            
            print(model_names)
            # wts=[0.7401,0.7506,0.6364,0.7484, 0.6725,0.7707,0.7219,0.7439,0.8003,0.7458]#[0.5651,0.5252,0.5537,0.5137,0.5744,0.4862,0.5255,0.5559,0.5755,0.5060]
        

        else:
            print("No SISA yet")

        def ensemble_evaluate(post_transforms, models):
            # print(post_transforms.transforms)
            if args.val==1:
                evaluator = EnsembleEvaluator(
                    device=device,
                    val_data_loader=test_loader, #test dataloader - this is loading all 5 sets of data
                    pred_keys=["pred"+str(i) for i in range(len(models))], 
                    networks=models, # models defined above
                    inferer=SlidingWindowInferer(
                        roi_size=(192,192, 144), sw_batch_size=4, overlap=0),
                    postprocessing=post_transforms, # this is going to call post_transforms based on type of ensemble
                    key_val_metric={
                            "test_mean_dice": MeanDice(
                                include_background=True,
                                output_transform=from_engine(["pred", "label"]) ,reduction='mean_channel' # takes all the preds and labels and turns them into one list each
                                
                            )},
                   
                    additional_metrics={ 
                        "Channelwise": MeanDice(
                        include_background=True,
                        output_transform=from_engine(["pred", "label"]),
                        reduction="mean_batch")
                    }
                )
            else:
                evaluator = EnsembleEvaluator(
                    device=device,
                    val_data_loader=test_loader, #test dataloader - this is loading all 5 sets of data
                    pred_keys=["pred"+str(i) for i in range(len(models))], 
                    networks=models, # models defined above
                    inferer=SlidingWindowInferer(
                        roi_size=(192,192, 144), sw_batch_size=4, overlap=0),
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
            mean_dice=evaluator.state.metrics['test_mean_dice']
            tumor_core=evaluator.state.metrics["Channelwise"][0]
            whole_tumor=evaluator.state.metrics["Channelwise"][1]
            enhancing_tumor=evaluator.state.metrics["Channelwise"][2]
            print("Mean Dice:",evaluator.state.metrics['test_mean_dice'],"metric_tc:",float(evaluator.state.metrics["Channelwise"][0]),"whole tumor:",float(evaluator.state.metrics["Channelwise"][1]),"enhancing tumor:",float(evaluator.state.metrics["Channelwise"][2]))#jbc
            
            return mean_dice,tumor_core,whole_tumor,enhancing_tumor
        
        
        

        
        
        model_num= len(model_names)
        mean_dice=[]
        tumor_core=[]
        whole_tumor=[]
        enhancing_tumor=[]
        scores={}
        
        if args.plot>0:
            for i in range(int(model_num)): #//5)):
                if args.plot==1:
                    if i%5!=0:
                     continue
                    model_steps=model_names#[:i+1]#[:(i+1)]
                    print(model_steps)  
                elif args.plot==2:
                    model_steps=[model_names[i]]
                    print(model_steps)                   
                    
                models=[]
                for name in model_steps:
                   
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
                    
                    model=torch.nn.DataParallel(model)
                    
                    model.load_state_dict(torch.load("/scratch/a.bip5/BraTS 2021/ssensemblemodels0922/Evaluation Folder/"+name),strict=False)
                    model.eval()
                    models.append(model)
                    
                num_models=len(models)  
                if args.avgmodel:
                    for key in model.state_dict().keys():
                        for i in models[:-1]:
                            model.state_dict()[key]=(i.state_dict()[key]+model.state_dict()[key] )
                        model.state_dict()[key]=model.state_dict()[key]/num_models
                    torch.save(
                        model.state_dict(),
                        os.path.join("./saved models", date.today().isoformat()+'T'+str(datetime.today().hour)+ args.model+"zoo_avg5160"))
                        
                    print("saved zoo model")
                    
                    args.ensemble = 0
                    args.load_name=date.today().isoformat()+'T'+str(datetime.today().hour)+ args.model+"zoo_avg"
                    
                    break
                        
                   
                                
      
                
                mean_post_transforms = Compose(
                    [
                        EnsureTyped(keys=["pred"+str(i) for i in range(len(models))]), #gives pred0..pred1...
                        ## SplitChanneld(keys=["pred"+str(i) for i in range(10)]),
                        
                        MeanEnsembled(
                            keys=["pred"+str(i) for i in range(len(models))], 
                            output_key="pred",
                          ##  # in this particular example, we use validation metrics as weights
                          ### weights=wts,
                        ),
                        Activationsd(keys="pred", sigmoid=True),
                        AsDiscreted(keys="pred", threshold=0.5),
                    ]
                ) 
                                   
                md,tc,wt,et=ensemble_evaluate(mean_post_transforms, models)
                if args.val==1:
                    scores[model_steps[0]]=md
                mean_dice.append(md.tolist())
                tumor_core.append(tc)
                whole_tumor.append(wt)
                enhancing_tumor.append(et)
                del models
                gc.collect()
                torch.cuda.empty_cache()
            # if args.val==1:
                    # sorted_scores=dict(sorted(scores.items(),key=lambda item: item[1])) # sorts the models by score
                    # print (sorted_scores)
            # print(mean_dice,'mean_dice')
            mean_dice_best=np.array(mean_dice).max(axis=0)
            mean_dice_model=np.array(mean_dice).max(axis=1)
            actual_mean_dice=np.array(mean_dice).mean()
            print('actual_mean_dice',actual_mean_dice)
            print("the best average mean dice from best results is", mean_dice_best.mean())
            scores_df=pd.DataFrame(scores)
            scores_df.to_csv('eval_score'+date.today().isoformat()+'T'+str(datetime.today().hour)+ args.model+'.csv')
            fig, ax = plt.subplots(figsize=(10,6))
            
            ax.plot(mean_dice_model, label="Mean Dice")
            ax.set_ylim(0.6,0.95)
            ax.set_xlabel("Number of models used")
            ax.set_ylabel("Dice score")
            ax.plot(tumor_core, label='Tumor Core')
            ax.plot(whole_tumor, label='Whole tumor')
            ax.plot(enhancing_tumor,label='Enhancing tumor')
            ax.legend()
            ax.set_title("Dice score change with number of models")
            fig.tight_layout()
            
            plt.savefig("Dice"+ model_names[0]+str(args.plot))
            
            # print('mean_dice', mean_dice) 
        else:          
            models=[]
            for i,name in enumerate(model_names):
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
                
                model=torch.nn.DataParallel(model)
                
                model.load_state_dict(torch.load("./"+name),strict=False)
                model.eval()
                models.append(model)
                
            num_models=len(models)  
            if args.avgmodel:
                for key in model.state_dict().keys():
                    for i in models[:-1]:
                        model.state_dict()[key]=(i.state_dict()[key]+model.state_dict()[key] )
                    model.state_dict()[key]=model.state_dict()[key]/num_models
                torch.save(
                    model.state_dict(),
                    os.path.join("./saved models", date.today().isoformat()+'T'+str(datetime.today().hour)+ args.model+"zoo_avg"))
                    
                print("saved zoo model")
                
                args.ensemble = 0
                args.load_name=date.today().isoformat()+'T'+str(datetime.today().hour)+ args.model+"zoo_avg"
     
                                

            mean_post_transforms = Compose(
                [
                    EnsureTyped(keys=["pred"+str(i) for i in range(len(models))]), #gives pred0..pred1...
                    ## SplitChanneld(keys=["pred"+str(i) for i in range(10)]),
                    
                    MeanEnsembled(
                        keys=["pred"+str(i) for i in range(len(models))], 
                        output_key="pred",
                      ##  # in this particular example, we use validation metrics as weights
                      ### weights=wts,
                    ),
                    Activationsd(keys="pred", sigmoid=True),
                    AsDiscreted(keys="pred", threshold=0.5),
                ]
            ) 
                               
            md,tc,wt,et=ensemble_evaluate(mean_post_transforms, models)
            mean_dice.append(md)
            tumor_core.append(tc)
            whole_tumor.append(wt)
            enhancing_tumor.append(et)
    elif args.ensemble==0:
        
        model.load_state_dict(torch.load("./ssensemblemodels0922/Evaluation Folder/"+args.load_name),strict=False)
        # model.to(device)
        model.eval()

        with torch.no_grad():

            for test_data in test_loader: # each image
                test_inputs = test_data["image"].to(device) # pass to gpu
                test_labels=test_data["label"].to(device)
                test_data["pred"] = sliding_window_inference(
                    inputs=test_inputs,
                    roi_size=(192,192, 144),
                    sw_batch_size=args.batch_size,
                    predictor=model,
                    overlap=0,
                )#inference(test_inputs) #perform inference
                #print(test_data["pred"].shape)
                test_data=[post_transforms(i) for i in decollate_batch(test_data)] #reverse the transform and get sigmoid then binarise
                test_outputs, test_labels =  from_engine(["pred", "label"])(test_data) # create list of images and labels
              
                
                #print("test outputs",test_outputs[0].shape)
                test_outputs=[i.to(device) for i in test_outputs]
                test_labels=[i.to(device) for i in test_labels]
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
        print(f"metric_tc: {metric_tc:.4f}", f"   metric_wt: {metric_wt:.4f}", f"   metric_et: {metric_et:.4f}")
     




