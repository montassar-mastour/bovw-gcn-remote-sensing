"""Utilities for handling sparse tensors."""
import torch
import numpy as np
from typing import Tuple


def save_sparse_tensor(tensor: torch.Tensor, filename: str):
    """
    Save sparse tensor to file.
    
    Args:
        tensor: Sparse PyTorch tensor
        filename: Output filename (.npz)
    """
    if not tensor.is_sparse:
        raise ValueError("Tensor must be sparse")
    
    np.savez_compressed(
        filename,
        indices=tensor._indices().cpu().numpy(),
        values=tensor._values().cpu().numpy(),
        size=tensor.size()
    )


def load_sparse_tensor(
    filename: str,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Load sparse tensor from file.
    
    Args:
        filename: Input filename (.npz)
        device: Device to load tensor on
        
    Returns:
        Sparse PyTorch tensor
    """
    data = np.load(filename)
    
    tensor = torch.sparse_coo_tensor(
        indices=torch.tensor(data['indices'], dtype=torch.long),
        values=torch.tensor(data['values'], dtype=torch.float32),
        size=tuple(data['size'])
    ).to(device)
    
    return tensor


def load_tensor_pair(
    filepath_prefix: str,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load Q and A sparse tensors.
    
    Args:
        filepath_prefix: Path prefix (without _Q.npz or _A.npz)
        device: Device to load tensors on
        
    Returns:
        Tuple of (Q, A) sparse tensors
    """
    Q = load_sparse_tensor(f"{filepath_prefix}_Q.npz", device)
    A = load_sparse_tensor(f"{filepath_prefix}_A.npz", device)
    
    return Q, A