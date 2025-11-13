"""
Script to preprocess dataset: generate superpixels and adjacency matrices.

Usage:
    python scripts/01_preprocess_data.py --config config/config.yaml
"""
import argparse
import os
import sys
import time
from pathlib import Path
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from data.dataset import get_dataloaders
from features.superpixel import SLICC
from features.graph_construction import construct_adjacency_matrix
from utils.sparse_utils import save_sparse_tensor
from utils.logger import setup_logger


def preprocess_split(
    data_loader,
    output_dir: str,
    config,
    logger,
    split_name: str
):
    """Preprocess a data split."""
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Processing {split_name} split...")
    start_time = time.time()
    
    for idx, (images, labels) in enumerate(tqdm(data_loader, desc=f"Preprocessing {split_name}")):
        image = images[0].numpy()  # Get single image
        label = labels.numpy()
        
        # Get filename
        full_dataset = data_loader.dataset.dataset
        indices = data_loader.dataset.indices
        image_idx = indices[idx]
        filename = os.path.basename(full_dataset.samples[image_idx][0])
        filename_base = os.path.splitext(filename)[0]
        
        # Generate superpixels
        slicc = SLICC(
            image,
            label,
            n_segments=config.superpixel.n_segments,
            compactness=config.superpixel.compactness,
            sigma=config.superpixel.sigma,
            min_size_factor=config.superpixel.min_size_factor,
            max_size_factor=config.superpixel.max_size_factor
        )
        
        Q, S = slicc.get_Q_and_S_and_Segments()
        
        # Construct adjacency matrix
        A = construct_adjacency_matrix(
            S,
            sigma=config.graph.sigma,
            distance_threshold=config.graph.distance_threshold,
            feature_threshold=config.graph.feature_threshold,
            k_neighbors=config.graph.k_neighbors
        )
        
        # Convert to sparse tensors
        from scipy.sparse import csr_matrix
        
        Q_sparse = csr_matrix(Q)
        A_sparse = csr_matrix(A)
        
        Q_indices = torch.tensor(list(zip(*Q_sparse.nonzero())), dtype=torch.long).T
        A_indices = torch.tensor(list(zip(*A_sparse.nonzero())), dtype=torch.long).T
        
        Q_torch = torch.sparse_coo_tensor(
            Q_indices,
            torch.tensor(Q_sparse.data, dtype=torch.float32),
            Q_sparse.shape
        )
        
        A_torch = torch.sparse_coo_tensor(
            A_indices,
            torch.tensor(A_sparse.data, dtype=torch.float32),
            A_sparse.shape
        )
        
        # Save tensors
        save_sparse_tensor(Q_torch, os.path.join(output_dir, f"{filename_base}_Q.npz"))
        save_sparse_tensor(A_torch, os.path.join(output_dir, f"{filename_base}_A.npz"))
        
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(data_loader)} images")
    
    elapsed = time.time() - start_time
    logger.info(f"{split_name} split completed in {elapsed/60:.2f} minutes")


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--max_samples', type=int, default=None,
                    help='Optional: limit number of images for quick testing')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logger
    logger = setup_logger(
        'preprocessing',
        log_file=os.path.join(config.paths.logs, 'preprocessing.log')
    )
    
    logger.info("Starting data preprocessing...")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load data
    train_loader, val_loader, test_loader, dataset_wrapper = get_dataloaders(
        dataset_root=config.dataset.root,
        dataset_name=config.dataset.name,
        image_size=config.dataset.image_size,
        batch_size=1,  # Process one image at a time
        train_split=config.dataset.train_split,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        num_workers=config.dataloader.num_workers,
        max_samples=args.max_samples,
    )
    if args.max_samples:
        logger.info(f"⚙️ Limiting preprocessing to first {args.max_samples} samples")
    
    # Process each split
    for split_name, loader in [
        ('train', train_loader),
        ('val', val_loader),
        ('test', test_loader)
    ]:
        output_dir = os.path.join(args.output_dir, split_name)
        preprocess_split(loader, output_dir, config, logger, split_name)
    
    logger.info("✓ Preprocessing complete!")


if __name__ == '__main__':
    main()