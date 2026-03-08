"""
1. Preprocessing Module
=======================
Loads DICOM images and prepares them for the neural network.

Pipeline:
    DICOM files → Load → Normalize → Resample → 3D Volume
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import pydicom
import SimpleITK as sitk

from config import HU_MIN, HU_MAX, TARGET_SPACING


def load_dicom_series(dicom_folder: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load a DICOM series from a folder.
    
    Args:
        dicom_folder: Path to folder containing .dcm files
        
    Returns:
        volume: 3D numpy array (Z, Y, X)
        properties: Dict with spacing, origin, direction
    """
    print(f"Loading DICOM from: {dicom_folder}")
    
    # Read DICOM series using SimpleITK
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_folder))
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_folder}")
    
    reader.SetFileNames(dicom_files)
    image_sitk = reader.Execute()
    
    # Convert to numpy array
    volume = sitk.GetArrayFromImage(image_sitk)  # Shape: (Z, Y, X)
    
    # Get metadata
    properties = {
        'spacing': np.array(image_sitk.GetSpacing()[::-1]),  # (Z, Y, X)
        'origin': np.array(image_sitk.GetOrigin()),
        'direction': np.array(image_sitk.GetDirection()).reshape(3, 3),
        'original_shape': volume.shape,
    }
    
    print(f"  Loaded shape: {volume.shape}")
    print(f"  Spacing: {properties['spacing']} mm")
    
    return volume, properties


def normalize_ct(volume: np.ndarray) -> np.ndarray:
    """
    Normalize CT Hounsfield units to [0, 1] range.
    
    Steps:
        1. Clip values to [HU_MIN, HU_MAX] (focus on blood vessels)
        2. Scale to [0, 1]
    
    Args:
        volume: 3D CT volume
        
    Returns:
        Normalized volume in [0, 1] range
    """
    print(f"Normalizing CT: [{volume.min():.0f}, {volume.max():.0f}] HU")
    
    # Step 1: Clip to focus range
    volume_clipped = np.clip(volume, HU_MIN, HU_MAX)
    
    # Step 2: Scale to [0, 1]
    volume_norm = (volume_clipped - HU_MIN) / (HU_MAX - HU_MIN)
    
    print(f"  Result: [{volume_norm.min():.3f}, {volume_norm.max():.3f}]")
    
    return volume_norm.astype(np.float32)


def resample_volume(volume: np.ndarray, 
                    original_spacing: np.ndarray,
                    target_spacing: Tuple[float, float, float] = TARGET_SPACING) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample volume to target spacing using linear interpolation.
    
    Args:
        volume: 3D numpy array
        original_spacing: (Z, Y, X) spacing in mm
        target_spacing: Target spacing in mm
        
    Returns:
        Resampled volume, new spacing
    """
    print(f"Resampling: {original_spacing} → {target_spacing} mm")
    
    # Calculate new shape
    original_shape = np.array(volume.shape)
    scale_factors = original_spacing / np.array(target_spacing)
    new_shape = np.round(original_shape * scale_factors).astype(int)
    
    # Create SimpleITK image for resampling
    image_sitk = sitk.GetImageFromArray(volume)
    image_sitk.SetSpacing(original_spacing[::-1].tolist())  # SimpleITK uses (X, Y, Z)
    
    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing[::-1])  # (X, Y, Z)
    resampler.SetSize(new_shape[::-1].tolist())       # (X, Y, Z)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    
    resampled_sitk = resampler.Execute(image_sitk)
    resampled = sitk.GetArrayFromImage(resampled_sitk)
    
    print(f"  Result shape: {resampled.shape}")
    
    return resampled, np.array(target_spacing)


def preprocess(dicom_folder: Path) -> Tuple[np.ndarray, Dict]:
    """
    Full preprocessing pipeline.
    
    Steps:
        1. Load DICOM series
        2. Normalize CT values
        3. Resample to target spacing
        
    Args:
        dicom_folder: Path to DICOM folder
        
    Returns:
        Preprocessed 4D array (1, Z, Y, X) ready for model
        Properties dict with metadata
    """
    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Step 1: Load DICOM
    volume, properties = load_dicom_series(dicom_folder)
    
    # Step 2: Normalize
    volume_norm = normalize_ct(volume)
    
    # Step 3: Resample
    volume_resampled, new_spacing = resample_volume(
        volume_norm, 
        properties['spacing']
    )
    
    # Add channel dimension (C, Z, Y, X)
    data = volume_resampled[np.newaxis, ...]  # (1, Z, Y, X)
    
    # Update properties
    properties['spacing'] = new_spacing
    properties['preprocessed_shape'] = data.shape
    
    print("=" * 50)
    print(f"Preprocessing complete. Shape: {data.shape}")
    print("=" * 50)
    
    return data, properties


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python 1_preprocessing.py <dicom_folder>")
        sys.exit(1)
    
    dicom_path = Path(sys.argv[1])
    data, props = preprocess(dicom_path)
    
    print(f"\nOutput shape: {data.shape}")
    print(f"Ready for model input!")
