import os
import numpy as np
import matplotlib.pyplot as plt

def plot_zero(input_image,prediction, mask, output_path,job_id, sub_id):

    # Convert prediction and mask to numpy arrays if they are lists
    if isinstance(input_image, list):
        input_image = input_image[0].cpu().numpy()
    if isinstance(prediction, list):
        prediction = prediction[0].cpu().numpy()
    if isinstance(mask, list):
        mask = mask[0].cpu().numpy()
    # Ensure prediction and mask have the same shape and at least 3 dimensions (with optional batch and channel).
    assert prediction.shape == mask.shape, "Prediction and mask must have the same shape"
    print('prediction.shape',prediction.shape)
    # Remove singleton batch dimension if present (shape[0] == 1)
    if prediction.shape[0] == 1:
        num_dims = len(prediction.shape)
        if num_dims == 5: 
            prediction = prediction[0]
            mask = mask[0]
            input_image=input_image[0]
    
    
    num_channels = prediction.shape[0] if num_dims == 4 else 1  # Assume shape is (C, D, H, W) or (D, H, W)
    
    
    # Create a folder for this id in the output path
    output_dir = os.path.join(output_path, job_id, sub_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the plot dictionary to store slices with descriptive names
    plot_dict = {}
    
    # Loop through each channel if there are 4 dimensions
     # Loop through each channel if there are multiple channels
    for c in range(num_channels):
        # Select the appropriate channel if multiple; otherwise, work directly with the single-channel data
        sample_pred = prediction[c] if num_channels > 1 else prediction[0]
        sample = mask[c] if num_channels > 1 else mask[0]
        sample_inp = input_image[c] if num_channels > 1 else input_image[0]
        print(sample.shape,'sample.shape')
        # Loop over each of the 3 spatial dimensions: depth (axis=0), height (axis=1), width (axis=2)
        for axis, axis_name in enumerate(['sagittal', 'coronal', 'axial']):
            # Array to store counts of non-zero elements for each slice along the current axis
            if axis == 0:
                nonzero_counts = [np.count_nonzero(sample[d, :, :]) for d in range(sample.shape[0])]
            elif axis == 1:
                nonzero_counts = [np.count_nonzero(sample[:, h, :]) for h in range(sample.shape[1])]
            elif axis == 2:
                nonzero_counts = [np.count_nonzero(sample[:, :, w]) for w in range(sample.shape[2])]
            
            # Get the slice index with the maximum non-zero count for the current axis
            max_area_index = np.argmax(nonzero_counts)
            
            # Save the appropriate slice to the dictionary with a descriptive key
            if axis == 0:
                slice_img_pred = sample_pred[max_area_index, :, :]
                slice_img = sample[max_area_index, :, :]
                slice_img_inp = sample_inp[max_area_index, :, :]
            elif axis == 1:
                slice_img_pred = sample_pred[:, max_area_index, :]
                slice_img = sample[:, max_area_index, :]
                slice_img_inp = sample_inp[:, max_area_index, :]
            elif axis == 2:
                slice_img_pred = sample_pred[:, :, max_area_index]
                slice_img = sample[:, :, max_area_index]
                slice_img_inp = sample_inp[:, :, max_area_index]
            
            # Add slice to the dictionary
            plot_dict[f'channel{c}_{axis_name}_pred_slice{max_area_index}'] = slice_img_pred
            plot_dict[f'channel{c}_{axis_name}_slice{max_area_index}'] = slice_img
            plot_dict[f'channel{c}_{axis_name}_slice_input{max_area_index}'] = slice_img_inp
    
    print(f'plotting for {sub_id}')
    # Plot each item in plot_dict and save as an image file
    for key, image in plot_dict.items():
        plt.figure()
        plt.imshow(image, cmap='gray', aspect='auto')
        plt.axis('off')
        plt.title(f"{key} - {sub_id}")
        plt.savefig(os.path.join(output_dir, f"{key}.png"), bbox_inches='tight')
        plt.close()

# Usage example
# plot_zero(prediction, mask, "output/path", "sample_id")
