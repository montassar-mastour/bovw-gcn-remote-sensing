"""Combined ResNet-GCN architecture (CEGCN)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights

from .gcn import GCNLayer
from .attention import AttentionLayer


class CEGCN(nn.Module):
    """
    Combined CNN-GCN model for image classification.
    
    Architecture:
        - ResNet50 backbone for feature extraction
        - GCN layers for superpixel-level reasoning
        - Attention mechanism for aggregation
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        class_count: int,
        gcn_layers: int = 2,
        hidden_dims: list = None,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        """
        Initialize CEGCN model.
        
        Args:
            height: Input image height
            width: Input image width
            channels: Number of input channels
            class_count: Number of output classes
            gcn_layers: Number of GCN layers
            hidden_dims: Hidden dimensions for GCN layers
            dropout: Dropout probability
            use_attention: Whether to use attention aggregation
        """
        super(CEGCN, self).__init__()
        
        self.height = height
        self.width = width
        self.channels = channels
        self.class_count = class_count
        self.use_attention = use_attention
        
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # ResNet50 backbone
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:8])
        
        # 1x1 convolution to reduce channels
        self.conv1x1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        input_dim = hidden_dims[0]
        
        for i in range(gcn_layers):
            if i == 0:
                output_dim = hidden_dims[1] if len(hidden_dims) > 1 else 256
            else:
                output_dim = hidden_dims[2] if len(hidden_dims) > 2 else 128
            
            self.gcn_layers.append(GCNLayer(input_dim, output_dim, dropout))
            input_dim = output_dim
        
        # Attention layer
        if self.use_attention:
            self.attention = AttentionLayer(input_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, class_count),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        Q: torch.Tensor,
        A: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (batch_size, channels, height, width)
            Q: Assignment matrix (batch_size, num_pixels, num_superpixels)
            A: Adjacency matrix (batch_size, num_superpixels, num_superpixels)
            
        Returns:
            Class logits (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Ensure Q and A are dense
        if Q.is_sparse:
            Q = Q.to_dense()
        if A.is_sparse:
            A = A.to_dense()
        
        # ResNet feature extraction
        resnet_features = self.resnet(x)  # (batch_size, 2048, h/32, w/32)
        resnet_features = self.conv1x1(resnet_features)  # (batch_size, 512, h/32, w/32)
        
        # Upsample to original resolution
        resnet_features = F.interpolate(
            resnet_features,
            size=(self.height, self.width),
            mode='bilinear',
            align_corners=False
        )
        
        # Reshape for GCN: (batch_size, num_pixels, feature_dim)
        batch_size, channels, height, width = resnet_features.shape
        resnet_features = resnet_features.view(batch_size, channels, -1).permute(0, 2, 1)
        
        # Create identity and mask for adjacency
        I = torch.eye(A.shape[1], device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        mask = torch.ceil(A * 0.00001)
        
        # Apply Q to get superpixel features
        H = torch.bmm(Q.transpose(1, 2), resnet_features)  # (batch_size, num_superpixels, feature_dim)
        
        # Pass through GCN layers
        for gcn_layer in self.gcn_layers:
            H, A = gcn_layer(H, A, mask, I)
        
        # Project back to pixel space (optional)
        output = torch.bmm(Q, H)  # (batch_size, num_pixels, feature_dim)
        
        # Aggregate using attention or mean pooling
        if self.use_attention:
            output = self.attention(output)  # (batch_size, feature_dim)
        else:
            output = torch.mean(output, dim=1)  # (batch_size, feature_dim)
        
        # Classification
        output = self.classifier(output)  # (batch_size, num_classes)
        
        return output
    
    def get_feature_extractor(self):
        """Return model without classification head for feature extraction."""
        return nn.Sequential(
            self.resnet,
            self.conv1x1
        )