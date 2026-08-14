# IdentityPreprocessor.py
import numpy as np
from typing import Union, List, Tuple
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class IdentityPreprocessor(DefaultPreprocessor):
    """
    Custom preprocessor that SKIPS cropping, resampling, and normalization.
    Assumes data is already:
    - Cropped to pancreas + margin
    - Resampled to 1.0 x 1.0 x 1.0 mm
    - Intensity normalized (windowing applied)
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        if self.verbose:
            print("Using IdentityPreprocessor - NO cropping, resampling, or normalization will be applied!")

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        """
        Override to skip preprocessing steps. Only handles:
        - Transpose (for orientation)
        - Sampling foreground locations (for training)
        """
        if self.verbose:
            print("IdentityPreprocessor: Skipping cropping, resampling, and normalization")
        
        # Create copy to avoid modifying input
        data = data.astype(np.float32, copy=True)
        has_seg = seg is not None
        if has_seg:
            seg = np.copy(seg)
        
        # Apply transpose_forward (orientation fix - this is necessary)
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if has_seg:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        
        # Store properties (no cropping)
        properties['shape_before_cropping'] = data.shape[1:]
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]
        properties['bbox_used_for_cropping'] = [[0, data.shape[1]], [0, data.shape[2]], [0, data.shape[3]]]
        
        # NO resampling - keep original spacing
        # NO normalization - data is already normalized
        
        # Sample foreground locations if segmentation exists
        if has_seg:
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels
            
            if label_manager.has_ignore_label:
                collect_for_this.append([-1] + label_manager.all_labels)
            
            properties['class_locations'] = self._sample_foreground_locations(
                seg, collect_for_this, verbose=self.verbose
            )
            
            # Convert segmentation to appropriate dtype
            if np.max(seg) > 127:
                seg = seg.astype(np.int16)
            else:
                seg = seg.astype(np.int8)
        
        if self.verbose:
            print(f"IdentityPreprocessor: Output shape {data.shape}, spacing {properties.get('spacing', 'unknown')}")
        
        return data, seg, properties

    def _normalize(self, data: np.ndarray, seg: np.ndarray, configuration_manager: ConfigurationManager,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        """
        Override to skip normalization - data is already normalized.
        """
        if self.verbose:
            print("IdentityPreprocessor: Skipping normalization (data assumed already normalized)")
        return data