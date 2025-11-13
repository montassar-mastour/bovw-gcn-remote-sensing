"""Checkpoint management utilities."""
import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional
import glob


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    checkpoint_dir: str,
    filename: Optional[str] = None,
    **kwargs
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        checkpoint_dir: Directory to save checkpoint
        filename: Custom filename (default: checkpoint_epoch_{epoch}.pth)
        **kwargs: Additional items to save
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        **kwargs
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint on
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"  Accuracy: {checkpoint.get('accuracy', 'N/A'):.4f}")
    
    return checkpoint


class CheckpointManager:
    """Manage model checkpoints with automatic cleanup."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        best_metric: str = 'accuracy',
        mode: str = 'max'
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            best_metric: Metric to track for best model
            mode: 'max' or 'min' for best metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_metric = best_metric
        self.mode = mode
        
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_checkpoint = None
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ):
        """Save checkpoint and manage cleanup."""
        # Save regular checkpoint
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            checkpoint_dir=str(self.checkpoint_dir),
            **metrics,
            **kwargs
        )
        
        # Check if this is the best model
        current_value = metrics.get(self.best_metric, 0.0)
        is_best = (
            (self.mode == 'max' and current_value > self.best_value) or
            (self.mode == 'min' and current_value < self.best_value)
        )
        
        if is_best:
            self.best_value = current_value
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(model.state_dict(), best_path)
            self.best_checkpoint = str(best_path)
            print(f"★ New best model saved! {self.best_metric}: {current_value:.4f}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only the most recent."""
        checkpoints = sorted(
            glob.glob(str(self.checkpoint_dir / "checkpoint_epoch_*.pth")),
            key=os.path.getctime,
            reverse=True
        )
        
        # Keep only max_checkpoints most recent
        for old_checkpoint in checkpoints[self.max_checkpoints:]:
            os.remove(old_checkpoint)
            print(f"  Removed old checkpoint: {os.path.basename(old_checkpoint)}")