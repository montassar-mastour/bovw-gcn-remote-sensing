"""
Train Bag of Visual Words classifier.

Usage:
    python scripts/05_train_bovw.py --config config/config.yaml
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from models.bovw import BoVWClassifier
from utils.logger import setup_logger
from training.metrics import plot_confusion_matrix


def load_histograms(filepath: str):
    """Load histograms from file."""
    data = np.load(filepath)
    return data['histograms'], data['labels']


def main():
    parser = argparse.ArgumentParser(description="Train BoVW classifier")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--histograms', type=str, default='data/features',
                       help='Directory with histogram files')
    parser.add_argument('--output', type=str, default='checkpoints/bovw_classifier.pkl',
                       help='Output file for trained classifier')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logger
    logger = setup_logger(
        'bovw_training',
        log_file=os.path.join(config.paths.logs, 'bovw_training.log')
    )
    
    logger.info("Starting BoVW classifier training...")
    
    # Load histograms
    logger.info("Loading histograms...")
    train_file = os.path.join(args.histograms, 'histograms_train.npz')
    val_file = os.path.join(args.histograms, 'histograms_val.npz')
    test_file = os.path.join(args.histograms, 'histograms_test.npz')
    if not os.path.exists(train_file):
        logger.error(f"Training histograms not found: {train_file}")
        logger.info("Please run script 04_build_codebook.py first")
        return
    
    X_train, y_train = load_histograms(train_file)
    logger.info(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    X_val, y_val = None, None
    if os.path.exists(val_file):
        X_val, y_val = load_histograms(val_file)
        logger.info(f"Val: {X_val.shape[0]} samples")
    
    X_test, y_test = None, None
    if os.path.exists(test_file):
        X_test, y_test = load_histograms(test_file)
        logger.info(f"Test: {X_test.shape[0]} samples")
    
    # Initialize classifier
    logger.info(f"Initializing {config.bovw.classifier} classifier...")
    classifier = BoVWClassifier(
        classifier_type=config.bovw.classifier,
        classifier_params=config.bovw.classifier_params,
        random_state=config.project.seed
    )
    
    # Train classifier
    logger.info("Training classifier...")
    classifier.fit(X_train, y_train)
    
    # Evaluate on validation set
    if X_val is not None:
        logger.info("\n" + "="*70)
        logger.info("VALIDATION SET EVALUATION")
        logger.info("="*70)
        val_metrics = classifier.evaluate(X_val, y_val)
        
        # Plot confusion matrix
        val_cm_path = os.path.join(config.paths.outputs, 'confusion_matrix_val_bovw.png')
        plot_confusion_matrix(
            val_metrics['predictions'],
            y_val,
            save_path=val_cm_path,
            normalize=False
        )
        logger.info(f"Validation confusion matrix saved: {val_cm_path}")
    
    # Evaluate on test set
    if X_test is not None:
        logger.info("\n" + "="*70)
        logger.info("TEST SET EVALUATION")
        logger.info("="*70)
        test_metrics = classifier.evaluate(X_test, y_test)
        
        # Plot confusion matrix
        test_cm_path = os.path.join(config.paths.outputs, 'confusion_matrix_test_bovw.png')
        plot_confusion_matrix(
            test_metrics['predictions'],
            y_test,
            save_path=test_cm_path,
            normalize=False
        )
        logger.info(f"Test confusion matrix saved: {test_cm_path}")
        
        # Plot normalized confusion matrix
        test_cm_norm_path = os.path.join(config.paths.outputs, 'confusion_matrix_test_bovw_normalized.png')
        plot_confusion_matrix(
            test_metrics['predictions'],
            y_test,
            save_path=test_cm_norm_path,
            normalize=True
        )
        logger.info(f"Normalized test confusion matrix saved: {test_cm_norm_path}")
    
    # Save classifier
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    classifier.save(args.output)
    logger.info(f"\n✓ Classifier saved to {args.output}")
    
    # Feature importance (if available)
    importance = classifier.get_feature_importance()
    if importance is not None:
        importance_file = os.path.join(config.paths.outputs, 'feature_importance.npy')
        np.save(importance_file, importance)
        logger.info(f"Feature importance saved to {importance_file}")
        
        # Log top features
        top_k = 10
        top_indices = np.argsort(importance)[-top_k:][::-1]
        logger.info(f"\nTop {top_k} most important visual words:")
        for rank, idx in enumerate(top_indices, 1):
            logger.info(f"  {rank}. Visual word {idx}: {importance[idx]:.6f}")
    
    logger.info("\n✓ BoVW classifier training complete!")


if __name__ == '__main__':
    main()