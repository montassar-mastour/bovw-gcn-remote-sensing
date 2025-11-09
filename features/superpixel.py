"""Superpixel segmentation using SLIC."""
import numpy as np
import torch
from skimage.segmentation import slic
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class SLICC:
    """SLIC-based superpixel segmentation with preprocessing."""
    
    def __init__(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        n_segments: int = 1500,
        compactness: float = 10.0,
        max_iter: int = 20,
        sigma: float = 5.0,
        min_size_factor: float = 0.3,
        max_size_factor: float = 2.0
    ):
        """
        Initialize SLICC segmentation.
        
        Args:
            image: Input image (C, H, W) or (H, W, C)
            labels: Image labels
            n_segments: Approximate number of segments
            compactness: Balances color/space proximity
            max_iter: Maximum iterations
            sigma: Gaussian smoothing sigma
            min_size_factor: Minimum segment size factor 
            max_size_factor: Maximum segment size factor
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        self.labels = labels
        
        # Preprocess the image
        if len(image.shape) == 3:
            if image.shape[0] in [1, 3]:  # (C, H, W)
                channels, height, width = image.shape
                data = image.reshape(channels, height * width).T
                self.original_shape = (height, width, channels)
            else:  # (H, W, C)
                height, width, channels = image.shape
                data = image.reshape(height * width, channels)
                self.original_shape = (height, width, channels)
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")
        
        # Standardize features
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        self.data = data.reshape(self.original_shape)
    
    def get_Q_and_S_and_Segments(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate superpixels and compute Q and S matrices.
        
        Returns:
            Tuple of (Q, S) where:
            - Q: Assignment matrix (num_pixels x num_superpixels)
            - S: Superpixel feature matrix (num_superpixels x feature_dim)
        """
        img = self.data
        h, w, d = img.shape
        
        # Apply SLIC segmentation
        segments = slic(
            img,
            n_segments=self.n_segments,
            compactness=self.compactness,
            convert2lab=False,
            sigma=self.sigma,
            enforce_connectivity=True,
            min_size_factor=self.min_size_factor,
            max_size_factor=self.max_size_factor,
            slic_zero=False
        )
        
        # Relabel segments for uniqueness
        if segments.max() + 1 != len(np.unique(segments)):
            segments = self._relabel_segments(segments)
        
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        
        # Flatten segments and initialize matrices
        segments_flat = segments.ravel()
        S = np.zeros((superpixel_count, d), dtype=np.float32)
        Q = np.zeros((h * w, superpixel_count), dtype=np.float32)
        x = img.reshape(-1, d)
        
        # Compute superpixel features and assignment matrix
        for i in range(superpixel_count):
            idx = np.where(segments_flat == i)[0]
            count = len(idx)
            if count > 0:
                pixels = x[idx]
                superpixel = np.sum(pixels, axis=0) / count
                S[i] = superpixel
                Q[idx, i] = 1
        
        self.S = S
        self.Q = Q
        
        return Q, S
    
    def _relabel_segments(self, labels: np.ndarray) -> np.ndarray:
        """
        Relabel segments to ensure uniqueness and continuity.
        
        Args:
            labels: Segment labels array
            
        Returns:
            Relabeled segments array
        """
        labels = np.array(labels, np.int64)
        unique_labels = list(set(labels.ravel().tolist()))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        new_labels = np.vectorize(label_map.get)(labels)
        return new_labels
    
    def get_segments(self) -> np.ndarray:
        """Get segment labels."""
        return self.segments if hasattr(self, 'segments') else None
    
    def get_superpixel_count(self) -> int:
        """Get number of superpixels."""
        return self.superpixel_count if hasattr(self, 'superpixel_count') else 0


def generate_superpixels(
    image: torch.Tensor,
    n_segments: int = 1500,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Generate superpixels from an image tensor.
    
    Args:
        image: Input image tensor (C, H, W)
        n_segments: Number of superpixels
        **kwargs: Additional arguments for SLICC
        
    Returns:
        Tuple of (Q, S, segments) as tensors/arrays
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = image
    
    # Create dummy labels
    labels = np.zeros(1)
    
    # Generate superpixels
    slicc = SLICC(image_np, labels, n_segments=n_segments, **kwargs)
    Q, S = slicc.get_Q_and_S_and_Segments()
    segments = slicc.get_segments()
    
    # Convert to tensors
    Q_tensor = torch.from_numpy(Q).float()
    S_tensor = torch.from_numpy(S).float()
    
    return Q_tensor, S_tensor, segments