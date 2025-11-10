"""Feature extraction module for BoVW."""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class FeatureExtractor:
    """Extract features from trained GCN model."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str,
        layer_name: Optional[str] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            model: Trained model
            device: Device for extraction
            layer_name: Specific layer to extract from (None = before classifier)
        """
        self.model = model
        self.device = device
        self.layer_name = layer_name
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Hook for feature extraction
        self.features = None
        if layer_name:
            self._register_hook(layer_name)
    
    def _register_hook(self, layer_name: str):
        """Register forward hook to extract features."""
        def hook_fn(module, input, output):
            self.features = output.detach()
        
        # Find and register hook on the specified layer
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook_fn)
                break
    
    def extract_features(
        self,
        images: torch.Tensor,
        Q: torch.Tensor,
        A: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            images: Input images (batch_size, C, H, W)
            Q: Assignment matrices
            A: Adjacency matrices
            
        Returns:
            Extracted features
        """
        with torch.no_grad():
            if self.layer_name:
                # Extract from specific layer using hook
                _ = self.model(images, Q, A)
                features = self.features
            else:
                # Extract from before classifier
                # We need to modify forward pass to return features
                features = self._extract_gcn_features(images, Q, A)
        
        return features
    
    def _extract_gcn_features(
        self,
        x: torch.Tensor,
        Q: torch.Tensor,
        A: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features from GCN layers (before classification).
        
        This replicates the model's forward pass up to the GCN output.
        """
        batch_size = x.size(0)
        
        # Ensure Q and A are dense
        if Q.is_sparse:
            Q = Q.to_dense()
        if A.is_sparse:
            A = A.to_dense()
        
        # ResNet feature extraction
        resnet_features = self.model.resnet(x)
        resnet_features = self.model.conv1x1(resnet_features)
        
        # Upsample
        resnet_features = torch.nn.functional.interpolate(
            resnet_features,
            size=(self.model.height, self.model.width),
            mode='bilinear',
            align_corners=False
        )
        
        # Reshape
        batch_size, channels, height, width = resnet_features.shape
        resnet_features = resnet_features.view(batch_size, channels, -1).permute(0, 2, 1)
        
        # Create identity and mask
        I = torch.eye(A.shape[1], device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        mask = torch.ceil(A * 0.00001)
        
        # Apply Q to get superpixel features
        H = torch.bmm(Q.transpose(1, 2), resnet_features)
        
        # Pass through GCN layers
        for gcn_layer in self.model.gcn_layers:
            H, A = gcn_layer(H, A, mask, I)
        
        # H now contains superpixel-level features
        # Shape: (batch_size, num_superpixels, feature_dim)
        return H


def extract_and_save_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tensor_loader_fn,
    device: str,
    output_dir: str,
    index_to_filename: dict
) -> None:
    """
    Extract features from entire dataset and save.
    
    Args:
        model: Trained model
        dataloader: Data loader
        tensor_loader_fn: Function to load Q, A tensors
        device: Device for extraction
        output_dir: Output directory
        index_to_filename: Mapping from index to filename
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    extractor = FeatureExtractor(model, device)
    
    print(f"Extracting features to: {output_dir}")
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Load tensors
        batch_Q, batch_A = [], []
        batch_filenames = []
        
        for i in range(len(images)):
            img_idx = batch_idx * len(images) + i
            filename = index_to_filename[img_idx]
            batch_filenames.append(filename)
            
            Q, A = tensor_loader_fn(filename)
            batch_Q.append(Q.to(device))
            batch_A.append(A.to(device))
        
        Q = torch.stack(batch_Q)
        A = torch.stack(batch_A)
        
        # Move to device
        images = images.to(device)
        
        # Extract features
        features = extractor.extract_features(images, Q, A)
        
        # Save each image's features
        for i, filename in enumerate(batch_filenames):
            feature = features[i].cpu().numpy()  # (num_superpixels, feature_dim)
            
            # Save as compressed numpy
            filename_base = Path(filename).stem
            save_path = Path(output_dir) / f"{filename_base}_features.npz"
            np.savez_compressed(save_path, features=feature, label=labels[i].numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Cleanup
        del Q, A, images, features
        torch.cuda.empty_cache()
    
    print(f"✓ Feature extraction complete!")