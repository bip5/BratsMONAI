 import os
import numpy as np
import re
import nibabel as nb
from numpy import save , load
import matplotlib.pyplot as plt
import glob
import cv2
import argparse
import sys

parser=argparse.ArgumentParser(description="shell options make life simpler sometimes")

parser.add_argument('--train_folder',default='/scratch/a.bip5/BraTS 2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData/*',type=str)

args=parser.parse_args()

#data dimensions (197,233,189 or sth like that)
all_files_train=glob.glob(args.train_folder)
all_files_test=glob.glob("/scratch/a.bip5/BraTS 2021/RSNA_ASNR_MICCAI_BraTS2021_TestData/*")


all_sizes=dict()
all_areasSagittal=dict()
all_areasFrontal=dict()
all_areasAxial=dict()
size_factors=dict()
dice_profSagittal=[]
dice_profFrontal=[]
dice_profAxial=[]
reg_score_sagittal=[]
reg_score_frontal=[]
reg_score_axial=[]
da_profSagittal=[]
da_profFrontal=[]
da_profAxial=[]
prev_area=0

segmentation=[]
kernel=np.ones((3,3),np.uint8)
for i in all_files_train:
    segmentation=glob.glob(i+'/*seg*') # look for all files with seg in the name
    if segmentation:
        mask=nb.load(segmentation[0]).get_fdata()#segmentation[0] gives the path
        mask_base=np.zeros_like(mask)
        lot_size=len(np.nonzero(mask)[0])
        whole_tumor=np.where(mask,1,mask_base)
        enhancing_tumor=np.where(mask==4,1,mask_base)
        tumor_core=np.where(mask==1,1,mask_base)                                             ##### TUMOUR SIZES
       
        
        area_storeSagittal=[]
        area_storeFrontal=[]
        area_storeAxial=[]
        areas=[]        
        diceP_indSagittal=[]
        diceP_indFrontal=[]
        diceP_indAxial=[]
        reg_indSagittal=[]
        reg_indFrontal=[]
        reg_indAxial=[]
        daP_indSagittal=[]
        daP_indFrontal=[]
        daP_indAxial=[]
    
        mask_bi=np.where(mask,1,mask)
        for x in range(mask.shape[0]):
            area_slice=len(np.nonzero(mask[x,:,:])[0])
            area_storeSagittal.append(area_slice)
            perimeter=len(np.nonzero(cv2.erode(mask[x,:,:],kernel))[0])
            rep_rad=np.sqrt(area_slice/np.pi)
            reg_score=perimeter/(2*np.pi*rep_rad + 0.001)                                       ##### PROFILES
            reg_indSagittal.append(reg_score)
            
            if x >0:
                dice=(2*(mask_bi[x-1,:,:]*mask_bi[x,:,:]).sum())\
                            /(mask_bi[x-1,:,:].sum()+mask_bi[x,:,:].sum()+0.001)
                diceP_indSagittal.append(dice)
                
                prev_area=len(np.nonzero(mask[x-1,:,:])[0])
                darea=2*abs(area_slice-prev_area)/(area_slice+prev_area+0.001)
                daP_indSagittal.append(darea)
                    
        for y in range(mask.shape[1]):
            area_slice=len(np.nonzero(mask[:,y,:])[0])
            area_storeFrontal.append(area_slice)
            perimeter=len(np.nonzero(cv2.erode(mask[:,y,:],kernel))[0])
            rep_rad=np.sqrt(area_slice/np.pi)
            reg_score=perimeter/(2*np.pi*rep_rad + 0.001)
            reg_indFrontal.append(reg_score)
            if y>0:
                dice=(2*(mask_bi[:,y-1,:]*mask_bi[:,y,:]).sum())\
                    /(mask_bi[:,y-1,:].sum()+mask_bi[:,y,:].sum()+0.001)
                diceP_indFrontal.append(dice)
                
                prev_area=len(np.nonzero(mask[:,y-1,:])[0])
                darea=2*abs(area_slice-prev_area)/(area_slice+prev_area+0.001)
                daP_indFrontal.append(darea)
                    
        for z in range(mask.shape[2]):
            area_slice=len(np.nonzero(mask[:,:,z])[0])
            # print(area_slice,'area_slice')
            area_storeAxial.append(area_slice) 
            perimeter=len(np.nonzero(cv2.erode(mask[:,:,z],kernel))[0])
            rep_rad=np.sqrt(area_slice/np.pi)
            reg_score=perimeter/(2*np.pi*rep_rad + 0.001)
            reg_indAxial.append(reg_score)
            if z>0:
                dice=(2*(mask_bi[:,:,z-1]*mask_bi[:,:,z]).sum())\
                    /(mask_bi[:,:,z-1].sum()+mask_bi[:,:,z].sum()+0.001)
                diceP_indAxial.append(dice)
                
                prev_area=len(np.nonzero(mask[:,:,z-1])[0])
                darea=2*abs(area_slice-prev_area)/(area_slice+prev_area+0.001)
                daP_indAxial.append(darea)
       
        all_areasSagittal[i[-5:]]=area_storeSagittal # areas in all axes
        all_areasFrontal[i[-5:]]=area_storeFrontal
        all_areasAxial[i[-5:]]=area_storeAxial
        all_sizes[i[-5:]]=lot_size
        
        dice_profSagittal.append(diceP_indSagittal)
        dice_profFrontal.append(diceP_indFrontal)
        dice_profAxial.append(diceP_indAxial)
        reg_score_axial.append(reg_indAxial)
        reg_score_frontal.append(reg_indFrontal)
        reg_score_sagittal.append(reg_indSagittal)
        da_profSagittal.append(daP_indSagittal)
        da_profAxial.append(daP_indAxial)
        da_profFrontal.append(daP_indFrontal)
            
    

            
for i in all_files_test:
    segmentation=glob.glob(i+'/*seg*') # look for all files with seg in the name
    if segmentation:
        mask=nb.load(segmentation[0]).get_fdata()
        mask_base=np.zeros_like(mask)
        lot_size=len(np.nonzero(mask)[0])
        whole_tumor=np.where(mask,1,mask_base)
        enhancing_tumor=np.where(mask==4,1,mask_base)
        tumor_core=np.where(mask==1,1,mask_base)
       
        
        area_storeSagittal=[]
        area_storeFrontal=[]
        area_storeAxial=[]
        areas=[]        
        diceP_indSagittal=[]
        diceP_indFrontal=[]
        diceP_indAxial=[]
        reg_indSagittal=[]
        reg_indFrontal=[]
        reg_indAxial=[]
        daP_indSagittal=[]
        daP_indFrontal=[]
        daP_indAxial=[]
        
        mask_bi=np.where(mask,1,mask)
        for x in range(mask.shape[0]):
            area_slice=len(np.nonzero(mask[x,:,:])[0])
            area_storeSagittal.append(area_slice)
            perimeter=len(np.nonzero(cv2.erode(mask[x,:,:],kernel))[0])
            rep_rad=np.sqrt(area_slice/np.pi)
            reg_score=perimeter/(2*np.pi*rep_rad + 0.001)
            reg_indSagittal.append(reg_score)
            
            if x >0:
                dice=(2*(mask_bi[x-1,:,:]*mask_bi[x,:,:]).sum())\
                            /(mask_bi[x-1,:,:].sum()+mask_bi[x,:,:].sum()+0.001)
                diceP_indSagittal.append(dice)
                
                prev_area=len(np.nonzero(mask[x-1,:,:])[0])
                darea=2*abs(area_slice-prev_area)/(area_slice+prev_area+0.001)
                daP_indSagittal.append(darea)
                    
        for y in range(mask.shape[1]):
            area_slice=len(np.nonzero(mask[:,y,:])[0])
            area_storeFrontal.append(area_slice)
            perimeter=len(np.nonzero(cv2.erode(mask[:,y,:],kernel))[0])
            rep_rad=np.sqrt(area_slice/np.pi)
            reg_score=perimeter/(2*np.pi*rep_rad + 0.001)
            reg_indFrontal.append(reg_score)
            if y>0:
                dice=(2*(mask_bi[:,y-1,:]*mask_bi[:,y,:]).sum())\
                    /(mask_bi[:,y-1,:].sum()+mask_bi[:,y,:].sum()+0.001)
                diceP_indFrontal.append(dice)
                
                prev_area=len(np.nonzero(mask[:,y-1,:])[0])
                darea=2*abs(area_slice-prev_area)/(area_slice+prev_area+0.001)
                daP_indFrontal.append(darea)
                    
        for z in range(mask.shape[2]):
            area_slice=len(np.nonzero(mask[:,:,z])[0])
            # print(area_slice,'area_slice')
            area_storeAxial.append(area_slice) 
            perimeter=len(np.nonzero(cv2.erode(mask[:,:,z],kernel))[0])
            rep_rad=np.sqrt(area_slice/np.pi)
            reg_score=perimeter/(2*np.pi*rep_rad + 0.001)
            reg_indAxial.append(reg_score)
            if z>0:
                dice=(2*(mask_bi[:,:,z-1]*mask_bi[:,:,z]).sum())\
                    /(mask_bi[:,:,z-1].sum()+mask_bi[:,:,z].sum()+0.001)
                diceP_indAxial.append(dice)
                
                prev_area=len(np.nonzero(mask[:,:,z-1])[0])
                darea=2*abs(area_slice-prev_area)/(area_slice+prev_area+0.001)
                daP_indAxial.append(darea)
       
        all_areasSagittal[i[-5:]]=area_storeSagittal # areas in all axes
        all_areasFrontal[i[-5:]]=area_storeFrontal
        all_areasAxial[i[-5:]]=area_storeAxial
        all_sizes[i[-5:]]=lot_size
        
        dice_profSagittal.append(diceP_indSagittal)
        dice_profFrontal.append(diceP_indFrontal)
        dice_profAxial.append(diceP_indAxial)
        reg_score_axial.append(reg_indAxial)
        reg_score_frontal.append(reg_indFrontal)
        reg_score_sagittal.append(reg_indSagittal)
        da_profSagittal.append(daP_indSagittal)
        da_profAxial.append(daP_indAxial)
        da_profFrontal.append(daP_indFrontal)
    
            
array_sizes=np.array(list(all_sizes.values()))
array_areasSagittal=np.array(list(all_areasSagittal.values()))

array_areasFrontal=np.array(list(all_areasFrontal.values()))
array_areasAxial=np.array(list(all_areasAxial.values()))

image_path='/scratch/a.bip5/BraTS 2021/ATLAS_R1.1 - Copy/train/images/031782_t1w_deface_stx.nii.gz'
image_sagittal=nb.load(image_path).get_fdata()[100,:,:]
image_frontal=nb.load(image_path).get_fdata()[:,100,:]
image_axial=nb.load(image_path).get_fdata()[:,:,100]

#Consecutive dice score
sagittal_profile=np.array(dice_profSagittal)
frontal_profile=np.array(dice_profFrontal)
axial_profile=np.array(dice_profAxial)

#consecutive reg score
sagittal_reg=np.array(reg_score_sagittal)
frontal_reg=np.array(reg_score_frontal)
axial_reg=np.array(reg_score_axial)

#delta area between slices
sagittal_da_profile=np.array(da_profSagittal)
frontal_da_profile=np.array(da_profFrontal)
axial_da_profile=np.array(da_profAxial)

plt.figure().set_figwidth(12)
plt.subplot(1,3,1)
plt.imshow(image_sagittal,cmap='Greys')
plt.xlabel('Sagittal view')
plt.subplot(1,3,2)
plt.imshow(image_frontal,cmap='Greys')
plt.xlabel('Frontal view')
plt.subplot(1,3,3)
plt.imshow(image_axial,cmap='Greys')
plt.xlabel('Axial view')
plt.savefig('00 Three view definition.png')

# print(array_sizes)
# print(array_sizes.shape,'array_sizes.shape')
# print(len(array_sizes))

#Distribution of overall size across patients
plt.figure()
plt.hist(array_sizes,10) 
plt.xlabel(r"Tumour Volume(mm^${3}$)")
plt.savefig('01 Tumour size.png')


plt.figure()
plt.subplot(3,1,1)
plt.plot(array_areasSagittal.mean(axis=0))
plt.xlabel('Sagittal mean area across subjects')
plt.subplot(3,1,2)
plt.plot(array_areasFrontal.mean(axis=0))
plt.xlabel('Frontal mean area across subjects')
plt.subplot(3,1,3)
plt.plot(array_areasAxial.mean(axis=0))
plt.xlabel('Axial mean area across subjects')
plt.tight_layout()
plt.savefig('02 Area across slices.png')

plt.figure().set_figwidth(12)
plt.subplot(1,3,1)
sagittal_area_dist=np.ma.masked_equal(array_areasSagittal,0).mean(axis=1)
plt.hist(sagittal_area_dist,10)
plt.xlabel(' Average area sagittal')
plt.subplot(1,3,2)
frontal_area_dist=np.ma.masked_equal(array_areasFrontal,0).mean(axis=1)
plt.hist(frontal_area_dist,10)
plt.xlabel('Average area frontal')
plt.subplot(1,3,3)
axial_area_dist=np.ma.masked_equal(array_areasAxial,0).mean(axis=1)
plt.hist(axial_area_dist,10)
plt.xlabel('Average area axial')
plt.tight_layout()
plt.savefig('03 Area distribution by view.png')

plt.figure()
#Getting a mean dice across patients across each slice
plt.subplot(3,1,1)
sagittal_reg_profile= sagittal_reg.mean(axis=0)
plt.plot(sagittal_reg_profile)
plt.xlabel('Sagittal shape factor profile')
plt.subplot(3,1,2)
frontal_reg_profile=frontal_reg.mean(axis=0)
plt.plot(frontal_reg_profile)
plt.xlabel('Frontal shape factor profile')
plt.subplot(3,1,3)
axial_reg_profile=axial_reg.mean(axis=0)
plt.plot(axial_reg_profile)
plt.xlabel('Axial shape factor profile')
plt.tight_layout()
plt.savefig('04 Expected shape factor across slices.png')

plt.figure().set_figwidth(12)
plt.subplot(1,3,1)
sagittal_reg_dist=np.ma.masked_equal(sagittal_reg,0).mean(axis=1)
plt.hist(sagittal_reg_dist,10)
plt.xlabel('Sagittal shape factor distribution')
plt.subplot(1,3,2)
frontal_reg_dist=np.ma.masked_equal(frontal_reg,0).mean(axis=1)
plt.hist(frontal_reg_dist,10)
plt.xlabel('Frontal shape factor distribution')
plt.subplot(1,3,3)
axial_reg_dist=np.ma.masked_equal(axial_reg,0).mean(axis=1)
plt.hist(axial_reg_dist,10)
plt.xlabel('Axial shape factor distribution')
plt.tight_layout()
plt.savefig('05 Shape factor distribution across subjects.png')
 
 
plt.figure()
mean_shape_factor_dist=(sagittal_reg_dist+frontal_reg_dist+axial_reg_dist)/3
plt.hist(mean_shape_factor_dist,10)
plt.xlabel('Overall shape factor distribution across subjects')
plt.savefig('06 Overall shape factor.png')


# plt.figure().set_figwidth(12)
# plt.subplot(1,2,1)
# plt.plot(array_areasSagittal)
# plt.xlabel("Tumour Area Sagittal")
# plt.subplot(1,2,2)
# plt.imshow(image_sagittal,cmap='Greys')
# plt.savefig("Tumour area Sagittal.png")
# plt.figure().set_figwidth(12)
# plt.subplot(1,2,1)
# plt.plot(array_areasFrontal)
# plt.xlabel("Tumour Area Frontal")
# plt.subplot(1,2,2)
# plt.imshow(image_frontal,cmap='Greys')
# plt.savefig("Tumour area Frontal.png")
# plt.figure().set_figwidth(12)
# plt.subplot(1,2,1)
# plt.plot(array_areasAxial)
# plt.xlabel("Tumour Area Axial")
# plt.subplot(1,2,2)
# plt.imshow(image_axial,cmap='Greys')
# plt.savefig("Tumour area Axial.png")



plt.figure()
#Getting a mean dice across patients across each slice
plt.subplot(3,1,1)
sagittal_mean_profile= sagittal_profile.mean(axis=0)
plt.plot(sagittal_mean_profile)
plt.xlabel('Sagittal mean dice profile across subjects')
plt.subplot(3,1,2)
frontal_mean_profile=frontal_profile.mean(axis=0)
plt.plot(frontal_mean_profile)
plt.xlabel('Sagittal mean dice profile across subjects')
plt.subplot(3,1,3)
axial_mean_profile=axial_profile.mean(axis=0)
plt.plot(axial_mean_profile)
plt.xlabel('Sagittal mean dice profile across subjects')
plt.tight_layout()
plt.savefig('07 Consecutive dice profile across subjects.png')


plt.figure().set_figwidth(12)
plt.subplot(1,3,1)
sagittal_profile_dist=np.ma.masked_equal(sagittal_profile,0).mean(axis=1)
plt.hist(sagittal_profile_dist,10)
plt.xlabel('Sagittal dice profile distribution')
plt.subplot(1,3,2)
frontal_profile_dist=np.ma.masked_equal(frontal_profile,0).mean(axis=1)
plt.hist(frontal_profile_dist,10)
plt.xlabel('Frontal dice profile distribution')
plt.subplot(1,3,3)
axial_profile_dist=np.ma.masked_equal(axial_profile,0).mean(axis=1)
plt.hist(axial_profile_dist,10)
plt.xlabel('Axial dice profile distribution')
plt.tight_layout()
plt.savefig('08 Overall consecutive dice profile across subjects.png')

plt.figure()
mean_dice_profile_dist=(sagittal_profile_dist+frontal_profile_dist+axial_profile_dist)/3
plt.hist(mean_dice_profile_dist,10)
plt.xlabel('Combined mean dice profile distribution')
plt.savefig('09 Combined mean dice profile distribution.png')

plt.figure()
#Getting a mean dice across patients across each slice
plt.subplot(3,1,1)
sagittal_mean_profile_da= sagittal_da_profile.mean(axis=0)
plt.plot(sagittal_mean_profile_da)
plt.xlabel('Sagittal mean delta area across subjects')
plt.subplot(3,1,2)
frontal_mean_profile_da=frontal_da_profile.mean(axis=0)
plt.plot(frontal_mean_profile_da)
plt.xlabel('Frontal mean delta area across subjects')
plt.subplot(3,1,3)
axial_mean_profile_da=axial_da_profile.mean(axis=0)
plt.plot(axial_mean_profile_da)
plt.xlabel('Axial mean delta area across subjects')
plt.tight_layout()
plt.savefig('10 Mean delta area across subjects.png')

plt.figure().set_figwidth(12)
plt.subplot(1,3,1)
sagittal_da_profile_dist=np.ma.masked_equal(sagittal_da_profile,0).mean(axis=1)
plt.hist(sagittal_da_profile_dist,10)
plt.xlabel('delta area sagittal')
plt.subplot(1,3,2)
frontal_da_profile_dist=np.ma.masked_equal(frontal_da_profile,0).mean(axis=1)
plt.hist(frontal_da_profile_dist,10)
plt.xlabel(' delta area  frontal ')
plt.subplot(1,3,3)
axial_da_profile_dist=np.ma.masked_equal(axial_da_profile,0).mean(axis=1)
plt.hist(axial_da_profile_dist,10)
plt.xlabel('delta area axial')
plt.savefig('11 delta area distribution.png')

plt.figure()
mean_da_profile_dist=(sagittal_da_profile_dist+frontal_da_profile_dist+axial_da_profile_dist)/3
plt.hist(mean_da_profile_dist,10)
plt.xlabel('Combined Mean delta area profile distribution')
plt.savefig('12 Combined Mean delta area profile distribution.png')
