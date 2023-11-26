import numpy as np
print(np.__version__)


from radiomics import featureextractor
import SimpleITK as sitk
from dataset import make_dataset
import pandas as pd
import datetime
import numpy as np


all1=make_dataset("/scratch/a.bip5/BraTS/BraTS_23_training")
        
gt_used=all1[1][:3]
imageall=all1[0][:3]
subject_features=dict()
feature_dict=dict()
results=[]
for image_paths,masks in zip(imageall,gt_used):
    for i,image_path in enumerate(image_paths):
        # Load the original image (assuming 'image_path' is the path to your MRI file)
        sub_id=masks[-30:-11]
        image = sitk.ReadImage(image_path)
        mask=sitk.ReadImage(masks)
        ######## Get the numpy array of the image
        # np_image = sitk.GetArrayFromImage(image)

        ############Desired mask size
        # mask_shape = (192, 192, 144)

        ######### Size of the space in which the mask will be centered
        # outer_shape = np_image.shape

        ###### Calculate the padding needed along each dimension
        # pad_z = (outer_shape[0] - mask_shape[0]) // 2
        # pad_y = (outer_shape[1] - mask_shape[1]) // 2
        # pad_x = (outer_shape[2] - mask_shape[2]) // 2

        ####### Calculate the indices where the mask will start and end in the larger space
        # start_z, end_z = pad_z, pad_z + mask_shape[0]
        # start_y, end_y = pad_y, pad_y + mask_shape[1]
        # start_x, end_x = pad_x, pad_x + mask_shape[2]

        ####### Create an array of zeros with the size of the outer space
        # full_mask = np.zeros(outer_shape, dtype=np.uint8)

        ######## Fill in the center of this array with ones to create the mask
        # full_mask[start_z:end_z, start_y:end_y, start_x:end_x] = 1

        ######### Convert the numpy array to a SimpleITK Image
        # mask_sitk = sitk.GetImageFromArray(full_mask)


        ########## Copy the information from the original image to the mask image
        # mask_sitk.CopyInformation(image)
        
    


        # Instantiate the extractor
        params = '/scratch/a.bip5/BraTS/scripts/Input/params.yaml'
        extractor = featureextractor.RadiomicsFeatureExtractor(params)

        # Extract features
        features = extractor.execute(image, mask)
        
        print(dir(features))

        # Print or analyze the extracted features
        for key, value in features.items():
            subject_features[f'{key}_{i}']=value #4 ins
        
        
    feature_dict[f'{sub_id}']=subject_features    
    results.append(subject_features)

now = datetime.datetime.now().strftime('%Y-%m-%d_%H')
df=pd.DataFrame(results)
df.to_csv(f'prd_feat_{now}.csv')