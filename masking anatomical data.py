import os
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from nilearn.image import new_img_like
from scipy.ndimage import zoom

# Define the folder path where your NIfTI files are located
folder_path = 'F:/team 16/new sort data/New folder/1'

# Define the operation you want to perform on each NIfTI file
def process_nifti_file(file_path):
    # Load the NIfTI image
    nifti_image = nib.load(file_path)
    
    # Get the data array from the image
    nifti_data = nifti_image.get_fdata()
    
    # Remove the last dimension (singleton dimension)
    nifti_data = np.squeeze(nifti_data)
    
    # Apply your desired operation on the data here
    
    # --- Example operation: Resample brain mask ---
    # Load the brain mask (modify this to load your specific mask)
    brain_mask = nib.load('working memory mask.nii')  # Replace with your brain mask file

    # Get the data array from the brain mask
    mask_data = brain_mask.get_fdata()

    # Ensure the dimensions match
    if nifti_data.shape != mask_data.shape:
        # Calculate the zoom factors for each axis
        zoom_factors = [n / m for n, m in zip(nifti_data.shape, mask_data.shape)]

        # Resample the brain mask to match the dimensions of the NIfTI image
        resampled_mask_data = zoom(mask_data, zoom_factors, order=0)  # Use order=0 for nearest-neighbor interpolation

        # Create a new NIfTI image from the resampled mask
        resampled_brain_mask = nib.Nifti1Image(resampled_mask_data, affine=nifti_image.affine)

        # Save the resampled brain mask
        nib.save(resampled_brain_mask, 'resampled_brain_mask.nii.gz')

        # Update the brain mask data
        mask_data = resampled_mask_data

    # Apply the brain mask to the modified NIfTI data
    masked_nifti_data = np.multiply(nifti_data, mask_data)

    # Create a new NIfTI image from the masked data
    masked_nifti_image = nib.Nifti1Image(masked_nifti_data, nifti_image.affine)

    # Define the output file path (you may need to adjust this based on your naming conventions)
    output_file = os.path.join(folder_path, 'masked_' + os.path.basename(file_path))

    # Save the modified NIfTI image to a new file
    nib.save(masked_nifti_image, output_file)

# Loop through all NIfTI files in the folder and apply the operation
for file_name in os.listdir(folder_path):
    if file_name.endswith('.nii'):  # Adjust the file extension as needed
        file_path = os.path.join(folder_path, file_name)
        process_nifti_file(file_path)


