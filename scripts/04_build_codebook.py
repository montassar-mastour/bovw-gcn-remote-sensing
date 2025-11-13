"""
Build visual vocabulary using K-Means clustering.

Usage:
    python scripts/04_build_codebook.py --config config/config.yaml --features data/features/train
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from features.codebook import CodebookGenerator, generate_histograms_from_features
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Build visual vocabulary")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--features', type=str, default='data/features/train',
                       help='Directory with extracted features')
    parser.add_argument('--output', type=str, default='checkpoints/cluster_centers.npy',
                       help='Output file for cluster centers')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples for fitting (None = all)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logger
    logger = setup_logger(
        'codebook',
        log_file=os.path.join(config.paths.logs, 'codebook.log')
    )
    
    logger.info("Starting codebook generation...")
    logger.info(f"Features directory: {args.features}")
    logger.info(f"Number of clusters: {config.bovw.num_clusters}")
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize codebook generator
    codebook = CodebookGenerator(
        n_clusters=config.bovw.num_clusters,
        batch_size=config.bovw.kmeans_batch_size,
        n_init=config.bovw.kmeans_n_init,
        random_state=config.project.seed,
        verbose=True
    )
    
    # Fit codebook
    logger.info("Fitting codebook (this may take a while)...")
    codebook.fit_incremental(args.features, max_samples=args.max_samples)
    
    # Save codebook
    codebook.save(args.output)
    logger.info(f"✓ Codebook saved to {args.output}")
    
    # Generate histograms for all splits
    logger.info("\nGenerating histograms for all splits...")
    
    features_base = Path(args.features).parent
    for split in ['train', 'val', 'test']:
        split_features_dir = features_base / split
        if not split_features_dir.exists():
            logger.warning(f"Split directory not found: {split_features_dir}")
            continue
        
        logger.info(f"Processing {split} split...")
        output_file = features_base / f"histograms_{split}.npz"
        
        generate_histograms_from_features(
            features_dir=str(split_features_dir),
            codebook=codebook,
            output_file=str(output_file)
        )
    
    logger.info("\n✓ Codebook generation complete!")
    logger.info(f"Cluster centers saved to: {args.output}")
    logger.info(f"Histograms saved to: {features_base}/histograms_*.npz")


if __name__ == '__main__':
    main()