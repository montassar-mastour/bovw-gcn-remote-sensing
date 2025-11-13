"""Evaluation utilities."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Callable, Optional
import numpy as np
from tqdm import tqdm

from utils.logger import get_logger
from .metrics import compute_metrics, plot_confusion_matrix, print_classification_report


class Evaluator:
    """Model evaluator."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str,
        class_names: Optional[list] = None,
        logger_name: str = 'evaluator'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            device: Device for evaluation
            class_names: List of class names
            logger_name: Logger name
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.logger = get_logger(logger_name)
    
    def evaluate(
        self,
        data_loader: DataLoader,
        tensor_loader_fn: Callable,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader
            tensor_loader_fn: Function to load Q, A tensors
            criterion: Loss function (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_outputs = []
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels, filenames in tqdm(data_loader, desc="Evaluating"):
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
                
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                
                # Track predictions
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_outputs.append(outputs.cpu())
                
                # Cleanup
                del Q, A, images, labels, outputs
                torch.cuda.empty_cache()
        
        # Compute metrics
        metrics = compute_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_targets),
            class_names=self.class_names
        )
        
        if criterion is not None:
            metrics['loss'] = total_loss / len(data_loader)
        
        # Log metrics
        self.logger.info("Evaluation Results:")
        for key, value in metrics.items():
            if not key.startswith('precision_') and not key.startswith('recall_') and not key.startswith('f1_'):
                self.logger.info(f"  {key}: {value:.4f}")
        
        # Store for confusion matrix
        self.last_predictions = np.array(all_predictions)
        self.last_targets = np.array(all_targets)
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = False
    ):
        """Plot confusion matrix from last evaluation."""
        if not hasattr(self, 'last_predictions'):
            raise RuntimeError("Must call evaluate() before plotting confusion matrix")
        
        plot_confusion_matrix(
            self.last_predictions,
            self.last_targets,
            class_names=self.class_names,
            save_path=save_path,
            normalize=normalize
        )
    
    def print_report(self):
        """Print classification report from last evaluation."""
        if not hasattr(self, 'last_predictions'):
            raise RuntimeError("Must call evaluate() before printing report")
        
        print_classification_report(
            self.last_predictions,
            self.last_targets,
            class_names=self.class_names
        )