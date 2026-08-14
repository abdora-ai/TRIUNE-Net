import nibabel as nib
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk

def preprocess_msd_pancreas(input_dir, output_dir, window=(-120, 240), margin=25, target_spacing=(1.0, 1.0, 1.0), 
                            do_resample=True, do_windowing=True, do_crop=True):
    """
    Apply paper's preprocessing to MSD Pancreas dataset
    """
    # Create output folders
    os.makedirs(os.path.join(output_dir, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labelsTr'), exist_ok=True)
    
    # Get all case files
    image_files = sorted(glob(os.path.join(input_dir, 'imagesTr', 'pancreas_*.nii.gz')))
    
    for img_path in tqdm(image_files, desc="Processing"):
        # Get case name
        case_name = os.path.basename(img_path).replace('.nii.gz', '')
        label_path = os.path.join(input_dir, 'labelsTr', f'{case_name}.nii.gz')
        
        if not os.path.exists(label_path):
            print(f"Warning: Label not found for {case_name}, skipping...")
            continue
        
        # Load image and label using SimpleITK (easier for resampling)
        img_sitk = sitk.ReadImage(img_path)
        label_sitk = sitk.ReadImage(label_path)
        
        # Convert to numpy for windowing
        img_data = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        label_data = sitk.GetArrayFromImage(label_sitk).astype(np.uint8)
        
        # Step 1: Windowing [-120, 240] HU
        if do_windowing:
            img_data = np.clip(img_data, window[0], window[1])
            img_data = (img_data - window[0]) / (window[1] - window[0])  # Normalize to [0,1]
        
        # Put back into SimpleITK for resampling
        img_sitk = sitk.GetImageFromArray(img_data)
        img_sitk.CopyInformation(sitk.ReadImage(img_path))  # Copy original metadata
        
        # Step 2: Resample to target spacing
        if do_resample:
            current_spacing = img_sitk.GetSpacing()
            if current_spacing != target_spacing:
                # Calculate new size
                old_size = img_sitk.GetSize()
                new_size = [int(round(old_size[i] * current_spacing[i] / target_spacing[i])) for i in range(3)]
                
                # Resample image (linear interpolation)
                resampler = sitk.ResampleImageFilter()
                resampler.SetSize(new_size)
                resampler.SetOutputSpacing(target_spacing)
                resampler.SetInterpolator(sitk.sitkLinear)
                img_resampled = resampler.Execute(img_sitk)
                
                # Resample label (nearest neighbor)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                label_resampled = resampler.Execute(label_sitk)
                
                img_data = sitk.GetArrayFromImage(img_resampled)
                label_data = sitk.GetArrayFromImage(label_resampled)
            else:
                img_data = sitk.GetArrayFromImage(img_sitk)
                label_data = sitk.GetArrayFromImage(label_sitk)
        
        # Step 3: Crop to pancreas + margin
        if do_crop:
            pancreas_mask = (label_data == 1) | (label_data == 2)
            coords = np.where(pancreas_mask)
            if len(coords[0]) > 0:
                z_min, y_min, x_min = np.min(coords, axis=1)
                z_max, y_max, x_max = np.max(coords, axis=1) + 1
                
                # Add margin
                z_min = max(0, z_min - margin)
                z_max = min(img_data.shape[0], z_max + margin)
                y_min = max(0, y_min - margin)
                y_max = min(img_data.shape[1], y_max + margin)
                x_min = max(0, x_min - margin)
                x_max = min(img_data.shape[2], x_max + margin)
                
                # Crop
                img_cropped = img_data[z_min:z_max, y_min:y_max, x_min:x_max]
                label_cropped = label_data[z_min:z_max, y_min:y_max, x_min:x_max]
            else:
                print(f"Warning: No pancreas found in {case_name}")
                img_cropped = img_data
                label_cropped = label_data
        else:
            img_cropped = img_data
            label_cropped = label_data
        
        # Save cropped files
        nib.save(nib.Nifti1Image(img_cropped, np.eye(4)), 
                 os.path.join(output_dir, 'imagesTr', f'{case_name}.nii.gz'))
        nib.save(nib.Nifti1Image(label_cropped, np.eye(4)), 
                 os.path.join(output_dir, 'labelsTr', f'{case_name}.nii.gz'))

# Usage
if __name__ == "__main__":
    input_dir = "/content/Task07_Pancreas"
    output_dir = "/content/Task07_Pancreas_Cropped"
    
    print("Starting preprocessing...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    preprocess_msd_pancreas(
        input_dir, 
        output_dir, 
        window=(-120, 240), 
        margin=25,
        do_resample=True,
        do_windowing=True,
        do_crop=True
    )
    
    print("Done!")