"""Attention mechanisms for aggregation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Attention-based aggregation layer."""
    
    def __init__(self, input_dim: int):
        """
        Initialize attention layer.
        
        Args:
            input_dim: Input feature dimension
        """
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)
    
    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Apply attention-based aggregation.
        
        Args:
            Y: Input features (batch_size, num_items, feature_dim)
            
        Returns:
            Aggregated features (batch_size, feature_dim)
        """
        # Compute attention weights
        weights = self.attention_weights(Y)  # (batch_size, num_items, 1)
        weights = torch.softmax(weights, dim=1)
        
        # Apply weighted sum
        output = torch.sum(weights * Y, dim=1)  # (batch_size, feature_dim)
        
        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            input_dim: Input feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Linear projections
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.out = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor (batch_size, seq_len, input_dim)
        """
        batch_size = x.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, V)
        
        # Concatenate heads and apply final linear
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.input_dim)
        output = self.out(output)
        
        return output