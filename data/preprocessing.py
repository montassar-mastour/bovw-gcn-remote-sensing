"""Data preprocessing utilities."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image data.
    
    Args:
        image: Image array with shape (C, H, W) or (H, W, C)
        
    Returns:
        Preprocessed image
    """
    # Ensure correct shape
    if image.shape[0] in [1, 3]:  # (C, H, W)
        channels, height, width = image.shape
        data = image.reshape(channels, height * width).T
    else:  # (H, W, C)
        height, width, channels = image.shape
        data = image.reshape(height * width, channels)
    
    # Standardize
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Reshape back
    if image.shape[0] in [1, 3]:
        return data.T.reshape(channels, height, width)
    else:
        return data.reshape(height, width, channels)


def get_image_stats(image: np.ndarray) -> dict:
    """Get image statistics."""
    return {
        'mean': np.mean(image, axis=(1, 2)) if len(image.shape) == 3 else np.mean(image),
        'std': np.std(image, axis=(1, 2)) if len(image.shape) == 3 else np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'shape': image.shape
    }