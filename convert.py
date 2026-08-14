import os
import re
from pathlib import Path

# Setup paths
base_path = Path('/content/Task07_Pancreas_Cropped')
imagesTr = base_path / 'imagesTr'
labelsTr = base_path / 'labelsTr'

# Rename images (add _0000 suffix)
print("Renaming images...")
for img in imagesTr.glob('pancreas_*.nii.gz'):
    num = int(re.search(r'(\d+)', img.name).group(1))
    new_name = f'pancreas_{num:04d}_0000.nii.gz'
    new_path = imagesTr / new_name
    img.rename(new_path)
    print(f"✓ {img.name} -> {new_name}")

# Rename labels (just pad with zeros, no suffix)
print("\nRenaming labels...")
for lbl in labelsTr.glob('pancreas_*.nii.gz'):
    num = int(re.search(r'(\d+)', lbl.name).group(1))
    new_name = f'pancreas_{num:04d}.nii.gz'
    new_path = labelsTr / new_name
    lbl.rename(new_path)
    print(f"✓ {lbl.name} -> {new_name}")

print(f"\n✅ Complete!")
print(f"Images: {len(list(imagesTr.glob('*.nii.gz')))} files")
print(f"Labels: {len(list(labelsTr.glob('*.nii.gz')))} files")