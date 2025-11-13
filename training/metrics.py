"""Metrics computation and visualization."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        if predictions.dim() > 1:  # Logits
            predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.cpu().numpy()
    
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Compute metrics
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Per-class metrics
    if class_names is not None:
        precision_per_class, recall_per_class, f1_per_class, _ = \
            precision_recall_fscore_support(targets, predictions, average=None, zero_division=0)
        
        for i, class_name in enumerate(class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
    
    return metrics


def plot_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    normalize: bool = False
):
    """
    Plot confusion matrix.
    
    Args:
        predictions: Predicted class indices
        targets: Ground truth labels
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize values
    """
    cm = confusion_matrix(targets, predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names or np.unique(targets),
        yticklabels=class_names or np.unique(targets),
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )
    
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
    
    plt.show()


def print_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None
):
    """Print detailed classification report."""
    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        digits=4
    )
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(report)