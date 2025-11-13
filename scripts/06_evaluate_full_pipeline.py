"""
Evaluate the complete pipeline end-to-end.

Usage:
    python scripts/06_evaluate_full_pipeline.py --config config/config.yaml
"""
import argparse
import os
import sys
from pathlib import Path
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from data.dataset import get_dataloaders
from models.resnet_gcn import CEGCN
from models.bovw import BoVWClassifier
from features.codebook import CodebookGenerator
from features.feature_extraction import FeatureExtractor
from utils.sparse_utils import load_tensor_pair
from utils.logger import setup_logger
from training.metrics import compute_metrics, plot_confusion_matrix, print_classification_report


class PipelineEvaluator:
    """End-to-end pipeline evaluator."""
    
    def __init__(
        self,
        gcn_model: torch.nn.Module,
        codebook: CodebookGenerator,
        bovw_classifier: BoVWClassifier,
        device: str,
        logger
    ):
        """Initialize pipeline evaluator."""
        self.gcn_model = gcn_model
        self.codebook = codebook
        self.bovw_classifier = bovw_classifier
        self.device = device
        self.logger = logger
        
        self.feature_extractor = FeatureExtractor(gcn_model, device)
    
    def evaluate_single_image(
        self,
        image: torch.Tensor,
        Q: torch.Tensor,
        A: torch.Tensor,
        true_label: int
    ) -> Dict[str, Any]:
        """
        Evaluate single image through complete pipeline.
        
        Returns:
            Dictionary with predictions and timing
        """
        timings = {}
        
        # 1. GCN Feature Extraction
        start = time.time()
        with torch.no_grad():
            gcn_features = self.feature_extractor.extract_features(
                image.unsqueeze(0),
                Q.unsqueeze(0),
                A.unsqueeze(0)
            )
            gcn_features = gcn_features.squeeze(0).cpu().numpy()
        timings['gcn_extraction'] = time.time() - start
        
        # 2. BoVW Histogram Generation
        start = time.time()
        histogram = self.codebook.transform(gcn_features)
        timings['histogram_generation'] = time.time() - start
        
        # 3. Classification
        start = time.time()
        prediction = self.bovw_classifier.predict(histogram.reshape(1, -1))[0]
        timings['classification'] = time.time() - start
        
        timings['total'] = sum(timings.values())
        
        return {
            'prediction': prediction,
            'true_label': true_label,
            'correct': prediction == true_label,
            'timings': timings,
            'histogram': histogram
        }
    
    def evaluate_dataset(
        self,
        dataloader,
        tensor_loader_fn,
        index_to_filename: dict,
        split_name: str = 'test'
    ) -> Dict[str, Any]:
        """Evaluate on entire dataset."""
        self.logger.info(f"\nEvaluating {split_name} set...")
        
        all_predictions = []
        all_labels = []
        all_timings = {'gcn_extraction': [], 'histogram_generation': [], 'classification': [], 'total': []}
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            for i in range(len(images)):
                # Get filename and load tensors
                img_idx = batch_idx * len(images) + i
                filename = index_to_filename[img_idx]
                Q, A = tensor_loader_fn(filename)
                
                # Evaluate single image
                result = self.evaluate_single_image(
                    images[i].to(self.device),
                    Q.to(self.device),
                    A.to(self.device),
                    labels[i].item()
                )
                
                all_predictions.append(result['prediction'])
                all_labels.append(result['true_label'])
                
                for key in all_timings:
                    all_timings[key].append(result['timings'][key])
            
            if (batch_idx + 1) % 10 == 0:
                self.logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Compute metrics
        metrics = compute_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_labels)
        )
        
        # Average timings
        avg_timings = {key: np.mean(values) for key, values in all_timings.items()}
        
        self.logger.info(f"\n{split_name.upper()} SET RESULTS:")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall: {metrics['recall']:.4f}")
        self.logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        
        self.logger.info(f"\nAverage Inference Time per Image:")
        self.logger.info(f"  GCN Extraction: {avg_timings['gcn_extraction']*1000:.2f} ms")
        self.logger.info(f"  Histogram: {avg_timings['histogram_generation']*1000:.2f} ms")
        self.logger.info(f"  Classification: {avg_timings['classification']*1000:.2f} ms")
        self.logger.info(f"  Total: {avg_timings['total']*1000:.2f} ms")
        self.logger.info(f"  Throughput: {1.0/avg_timings['total']:.2f} images/second")
        
        return {
            'metrics': metrics,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'timings': avg_timings
        }


def plot_comparison_results(
    results: Dict[str, Any],
    output_dir: str,
    logger
):
    """Plot comparison between different evaluation metrics."""
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Accuracy comparison (if multiple splits)
    ax = axes[0, 0]
    splits = list(results.keys())
    accuracies = [results[split]['metrics']['accuracy'] for split in splits]
    
    bars = ax.bar(splits, accuracies, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy by Split', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Timing breakdown
    ax = axes[0, 1]
    timing_categories = ['GCN\nExtraction', 'Histogram\nGeneration', 'Classification']
    test_timings = results['test']['timings']
    timing_values = [
        test_timings['gcn_extraction'] * 1000,
        test_timings['histogram_generation'] * 1000,
        test_timings['classification'] * 1000
    ]
    
    bars = ax.bar(timing_categories, timing_values, color=['#9b59b6', '#f39c12', '#1abc9c'])
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Inference Time Breakdown (Test Set)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Metrics comparison
    ax = axes[1, 0]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    test_metrics = results['test']['metrics']
    metrics_values = [
        test_metrics['accuracy'],
        test_metrics['precision'],
        test_metrics['recall'],
        test_metrics['f1_score']
    ]
    
    bars = ax.bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics (Test Set)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    PIPELINE EVALUATION SUMMARY
    {'='*40}
    
    Test Set Results:
    • Accuracy: {test_metrics['accuracy']:.4f}
    • Precision: {test_metrics['precision']:.4f}
    • Recall: {test_metrics['recall']:.4f}
    • F1-Score: {test_metrics['f1_score']:.4f}
    
    Performance:
    • Avg. Inference: {test_timings['total']*1000:.2f} ms
    • Throughput: {1.0/test_timings['total']:.1f} images/sec
    
    Model Configuration:
    • GCN Layers: 2
    • Visual Words: 1000
    • Classifier: Random Forest
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'pipeline_evaluation_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Evaluation summary saved: {output_path}")
    plt.close()


def create_index_to_filename_mapping(dataloader):
    """Create mapping from index to filename."""
    full_dataset = dataloader.dataset.dataset
    indices = dataloader.dataset.indices
    
    index_to_filename = {
        i: os.path.basename(full_dataset.samples[idx][0])
        for i, idx in enumerate(indices)
    }
    
    return index_to_filename


def get_tensor_loader_fn(processed_data_dir: str, split: str):
    """Create tensor loader function."""
    def loader_fn(filename):
        filepath = os.path.join(processed_data_dir, split, filename)
        filepath_base = os.path.splitext(filepath)[0]
        return load_tensor_pair(filepath_base)
    return loader_fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate full pipeline")
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--gcn_checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--codebook', type=str, default='checkpoints/cluster_centers.npy')
    parser.add_argument('--bovw_classifier', type=str, default='checkpoints/bovw_classifier.pkl')
    parser.add_argument('--processed_data', type=str, default='data/processed')
    parser.add_argument('--splits', type=str, nargs='+', default=['test'],
                       help='Splits to evaluate')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logger
    logger = setup_logger(
        'pipeline_evaluation',
        log_file=os.path.join(config.paths.logs, 'pipeline_evaluation.log')
    )
    
    logger.info("="*70)
    logger.info("FULL PIPELINE EVALUATION")
    logger.info("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config.device.use_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load GCN model
    logger.info("\nLoading GCN model...")
    gcn_model = CEGCN(
        height=config.dataset.image_size[0],
        width=config.dataset.image_size[1],
        channels=3,
        class_count=config.dataset.num_classes,
        gcn_layers=config.model.gcn_layers,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
        use_attention=config.model.use_attention
    ).to(device)
    
    checkpoint = torch.load(args.gcn_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        gcn_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        gcn_model.load_state_dict(checkpoint)
    
    gcn_model.eval()
    logger.info("✓ GCN model loaded")
    
    # Load codebook
    logger.info("Loading codebook...")
    codebook = CodebookGenerator(n_clusters=config.bovw.num_clusters)
    codebook.load(args.codebook)
    logger.info("✓ Codebook loaded")
    
    # Load BoVW classifier
    logger.info("Loading BoVW classifier...")
    bovw_classifier = BoVWClassifier()
    bovw_classifier.load(args.bovw_classifier)
    logger.info("✓ BoVW classifier loaded")
    
    # Load data
    logger.info("\nLoading dataset...")
    train_loader, val_loader, test_loader, dataset_wrapper = get_dataloaders(
        dataset_root=config.dataset.root,
        dataset_name=config.dataset.name,
        image_size=config.dataset.image_size,
        batch_size=1,  # Process one at a time for timing
        train_split=config.dataset.train_split,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        num_workers=config.dataloader.num_workers
    )
    
    # Create evaluator
    evaluator = PipelineEvaluator(
        gcn_model=gcn_model,
        codebook=codebook,
        bovw_classifier=bovw_classifier,
        device=str(device),
        logger=logger
    )
    
    # Evaluate each split
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    results = {}
    for split_name in args.splits:
        if split_name not in loaders:
            logger.warning(f"Unknown split: {split_name}, skipping...")
            continue
        
        loader = loaders[split_name]
        tensor_loader_fn = get_tensor_loader_fn(args.processed_data, split_name)
        index_to_filename = create_index_to_filename_mapping(loader)
        
        split_results = evaluator.evaluate_dataset(
            dataloader=loader,
            tensor_loader_fn=tensor_loader_fn,
            index_to_filename=index_to_filename,
            split_name=split_name
        )
        
        results[split_name] = split_results
        
        # Plot confusion matrix
        cm_path = os.path.join(config.paths.outputs, f'confusion_matrix_{split_name}_pipeline.png')
        plot_confusion_matrix(
            split_results['predictions'],
            split_results['labels'],
            class_names=dataset_wrapper.class_names,
            save_path=cm_path,
            normalize=False
        )
        
        # Print detailed report
        print_classification_report(
            split_results['predictions'],
            split_results['labels'],
            class_names=dataset_wrapper.class_names
        )
    
    # Create comparison plots
    if len(results) > 0:
        logger.info("\nGenerating comparison plots...")
        plot_comparison_results(results, config.paths.outputs, logger)
    
    # Save results
    results_file = os.path.join(config.paths.outputs, 'pipeline_evaluation_results.npz')
    save_data = {
        f'{split}_predictions': results[split]['predictions']
        for split in results
    }
    save_data.update({
        f'{split}_labels': results[split]['labels']
        for split in results
    })
    np.savez_compressed(results_file, **save_data)
    logger.info(f"✓ Results saved: {results_file}")
    
    logger.info("\n" + "="*70)
    logger.info("✓ PIPELINE EVALUATION COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    main()