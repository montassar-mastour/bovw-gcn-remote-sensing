"""Graph Convolutional Network layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
    
    
class GCNLayer(nn.Module):
    """Graph Convolutional Network layer with attention mechanism."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        """
        Initialize GCN layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            dropout: Dropout probability
        """
        super(GCNLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Normalization and activation
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.activation = nn.LeakyReLU()
        
        # Feature transformation layers
        self.linear_theta_1 = nn.Linear(input_dim, 256)
        self.activation_1 = nn.LeakyReLU()
        
        self.linear_theta_2 = nn.Linear(256, input_dim)
        self.activation_2 = nn.LeakyReLU()
        
        # Output transformation
        self.linear_out = nn.Linear(input_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        H: torch.Tensor,
        A: torch.Tensor,
        mask: torch.Tensor,
        I: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            H: Node features (batch_size, num_nodes, input_dim)
            A: Adjacency matrix (batch_size, num_nodes, num_nodes)
            mask: Mask for adjacency (batch_size, num_nodes, num_nodes)
            I: Identity matrix (batch_size, num_nodes, num_nodes)
            
        Returns:
            Tuple of (output features, updated adjacency matrix)
        """
        # Apply batch normalization
        H = self.batch_norm(H.permute(0, 2, 1)).permute(0, 2, 1)
        
        # First transformation
        H_theta_1 = self.linear_theta_1(H)
        H_theta_1 = self.activation_1(H_theta_1)
        
        # Second transformation
        H_theta_2 = self.linear_theta_2(H_theta_1)
        H_theta_2 = self.activation_2(H_theta_2)
        
        # Compute attention-based adjacency
        e = torch.bmm(H_theta_1, H_theta_1.transpose(1, 2))
        e = torch.sigmoid(e)
        
        # Apply mask and add identity
        zero_vec = -9e15 * torch.ones_like(e)
        A_updated = torch.where(mask > 0, e, zero_vec) + I
        A_updated = F.softmax(A_updated, dim=2)
        
        # Graph convolution
        H_out = self.linear_out(H)
        output = self.activation(torch.bmm(A_updated, H_out))
        
        # Apply dropout
        output = self.dropout(output)
        
        return output, A_updated
    
    def __repr__(self):
        return f"GCNLayer(input_dim={self.input_dim}, output_dim={self.output_dim})"


class SimpleGCNLayer(nn.Module):
    """Simplified GCN layer without attention."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        """Initialize simple GCN layer."""
        super(SimpleGCNLayer, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            H: Node features (batch_size, num_nodes, input_dim)
            A: Adjacency matrix (batch_size, num_nodes, num_nodes)
            
        Returns:
            Output features (batch_size, num_nodes, output_dim)
        """
        # Graph convolution: A @ H @ W
        H = torch.bmm(A, H)
        H = self.linear(H)
        H = self.activation(H)
        H = self.dropout(H)
        
        return H