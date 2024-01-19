import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/')

import numpy as np
print(np.__version__)


from radiomics import featureextractor
import SimpleITK as sitk
from Input.dataset import make_dataset
import pandas as pd
import datetime
import numpy as np


all1=make_dataset("/scratch/a.bip5/BraTS/BraTS_23_training")
        
gt_used=all1[1]
imageall=all1[0]

results=[]
feature_ids=[]
for image_paths,masks in zip(imageall,gt_used):
    subject_features=dict()
    
    sub_id=masks[-30:-11]
    for i,image_path in enumerate(image_paths):
        # Load the original image (assuming 'image_path' is the path to your MRI file)
        
        image = sitk.ReadImage(image_path)
        mask=sitk.ReadImage(masks)
        
        # Convert the mask to an integer type
        mask = sitk.Cast(mask, sitk.sitkUInt16)
        binary_mask = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
        non_zero_voxels = sitk.GetArrayFromImage(mask).nonzero()[0].size
        print(f"Mask {masks} has {non_zero_voxels} non-zero voxels")
            
        # Instantiate the extractor
        params = '/scratch/a.bip5/BraTS/scripts/Analysis/params.yaml'
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        
        # bb = sitk.LabelShapeStatisticsImageFilter()
        # bb.Execute(mask)
        # bounding_box = bb.GetBoundingBox(1)
        # print("LOCAL Bounding Box:", bounding_box)
        try:            
            features = extractor.execute(image, binary_mask)
        except:
            print('failed to extract features using', masks)
            
            # sys.exit()
            continue
        # print(dir(features))

        # Print or analyze the extracted features
        for key, value in features.items():
            subject_features[f'{key}_{i}']=value #4 ins
        
        
    feature_ids.append(sub_id)   
    results.append(subject_features)

now = datetime.datetime.now().strftime('%Y-%m-%d_%H')
df=pd.DataFrame(results,index=feature_ids)
df.to_csv(f'prd_feat_{now}.csv')