"""
Extract features from trained GCN model for BoVW.

Usage:
    python scripts/03_extract_features.py --config config/config.yaml --checkpoint checkpoints/best_model.pth
"""
import argparse
import os
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from data.dataset import get_dataloaders
from models.resnet_gcn import CEGCN
from features.feature_extraction import extract_and_save_features
from utils.sparse_utils import load_tensor_pair
from utils.logger import setup_logger


def get_tensor_loader_fn(processed_data_dir: str, split: str):
    """Create tensor loader function."""
    def loader_fn(filename):
        filepath = os.path.join(processed_data_dir, split, filename)
        filepath_base = os.path.splitext(filepath)[0]
        return load_tensor_pair(filepath_base)
    return loader_fn


def create_index_to_filename_mapping(dataloader):
    """Create mapping from index to filename."""
    full_dataset = dataloader.dataset.dataset
    indices = dataloader.dataset.indices
    
    index_to_filename = {
        i: os.path.basename(full_dataset.samples[idx][0])
        for i, idx in enumerate(indices)
    }
    
    return index_to_filename


def main():
    parser = argparse.ArgumentParser(description="Extract features for BoVW")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--processed_data', type=str, default='data/processed',
                       help='Directory with processed data')
    parser.add_argument('--output_dir', type=str, default='data/features',
                       help='Output directory for features')
    parser.add_argument('--splits', type=str, nargs='+', 
                       default=['train', 'val', 'test'],
                       help='Splits to process')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logger
    logger = setup_logger(
        'feature_extraction',
        log_file=os.path.join(config.paths.logs, 'feature_extraction.log')
    )
    
    logger.info("Starting feature extraction...")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config.device.use_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = CEGCN(
        height=config.dataset.image_size[0],
        width=config.dataset.image_size[1],
        channels=3,
        class_count=config.dataset.num_classes,
        gcn_layers=config.model.gcn_layers,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
        use_attention=config.model.use_attention
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("✓ Model loaded successfully")
    
    # Load data
    train_loader, val_loader, test_loader, dataset_wrapper = get_dataloaders(
        dataset_root=config.dataset.root,
        dataset_name=config.dataset.name,
        image_size=config.dataset.image_size,
        batch_size=config.dataloader.batch_size,
        train_split=config.dataset.train_split,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        num_workers=config.dataloader.num_workers
    )
    
    # Process each split
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    for split_name in args.splits:
        if split_name not in loaders:
            logger.warning(f"Unknown split: {split_name}, skipping...")
            continue
        
        logger.info(f"\nProcessing {split_name} split...")
        
        loader = loaders[split_name]
        tensor_loader_fn = get_tensor_loader_fn(args.processed_data, split_name)
        output_split_dir = os.path.join(args.output_dir, split_name)
        index_to_filename = create_index_to_filename_mapping(loader)
        
        extract_and_save_features(
            model=model,
            dataloader=loader,
            tensor_loader_fn=tensor_loader_fn,
            device=str(device),
            output_dir=output_split_dir,
            index_to_filename=index_to_filename
        )
        
        logger.info(f"✓ {split_name} split complete")
    
    logger.info("\n✓ Feature extraction complete for all splits!")


if __name__ == '__main__':
    main()