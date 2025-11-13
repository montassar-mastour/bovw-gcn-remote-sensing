"""Training utilities."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import time
from tqdm import tqdm

from utils.checkpoint import CheckpointManager
from utils.logger import get_logger
from .metrics import compute_metrics


class Trainer:
    """Model trainer with checkpoint management and logging."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        checkpoint_dir: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_checkpoints: int = 5,
        gradient_clip: Optional[float] = None,
        logger_name: str = 'trainer'
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            scheduler: Learning rate scheduler
            max_checkpoints: Maximum checkpoints to keep
            gradient_clip: Gradient clipping value
            logger_name: Logger name
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints,
            best_metric='accuracy',
            mode='max'
        )
        
        # Logger
        self.logger = get_logger(logger_name)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_accuracy = 0.0
        
        # History
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        tensor_loader_fn: Callable
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            tensor_loader_fn: Function to load Q, A tensors given filename
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, labels, filenames) in enumerate(progress_bar):
            # Load Q and A tensors
            batch_Q, batch_A = [], []
            for filename in filenames:
                Q, A = tensor_loader_fn(filename)
                batch_Q.append(Q.to(self.device))
                batch_A.append(A.to(self.device))
            
            Q = torch.stack(batch_Q)
            A = torch.stack(batch_A)
            
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, Q, A)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            self.global_step += 1
            
            # Cleanup
            del Q, A, images, labels, outputs, loss
            torch.cuda.empty_cache()
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        metrics = compute_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_targets)
        )
        
        metrics['loss'] = avg_loss
        
        self.logger.info(
            f"Train Epoch {self.current_epoch}: "
            f"Loss={avg_loss:.4f}, Accuracy={metrics['accuracy']:.4f}"
        )
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        tensor_loader_fn: Callable
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            tensor_loader_fn: Function to load Q, A tensors
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels, filenames in tqdm(val_loader, desc="Validation"):
                # Load tensors
                batch_Q, batch_A = [], []
                for filename in filenames:
                    Q, A = tensor_loader_fn(filename)
                    batch_Q.append(Q.to(self.device))
                    batch_A.append(A.to(self.device))
                
                Q = torch.stack(batch_Q)
                A = torch.stack(batch_A)
                
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images, Q, A)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                # Cleanup
                del Q, A, images, labels, outputs
                torch.cuda.empty_cache()
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_targets)
        )
        
        metrics['loss'] = avg_loss
        
        self.logger.info(
            f"Val Epoch {self.current_epoch}: "
            f"Loss={avg_loss:.4f}, Accuracy={metrics['accuracy']:.4f}"
        )
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tensor_loader_fn: Callable,
        val_tensor_loader_fn: Callable,
        num_epochs: int,
        start_epoch: int = 1
    ):
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            tensor_loader_fn: Function to load Q, A tensors
            num_epochs: Number of epochs
            start_epoch: Starting epoch number
        """
        self.current_epoch = start_epoch
        
        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, tensor_loader_fn)
            
            # Validate
            val_metrics = self.validate(val_loader, val_tensor_loader_fn)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=val_metrics
            )
            
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            
            print("-" * 80)

    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history['train_accuracy'], label='Train')
        ax2.plot(self.history['val_accuracy'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"✓ Training history saved: {save_path}")
        
        plt.show()