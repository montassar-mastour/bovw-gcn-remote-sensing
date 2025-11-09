"""
Script to train the ResNet-GCN model.

Usage:
    python scripts/02_train_gcn.py --config config/config.yaml
"""
import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from data.dataset import get_dataloaders
from models.resnet_gcn import CEGCN
from training.train import Trainer
from utils.sparse_utils import load_tensor_pair
from utils.logger import setup_logger


def get_tensor_loader_fn(processed_data_dir: str, split: str):
    """Create tensor loader function."""
    def loader_fn(filename):
        filepath = os.path.join(processed_data_dir, split, filename)
        filepath_base = os.path.splitext(filepath)[0]
        return load_tensor_pair(filepath_base)
    return loader_fn


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-GCN model")
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--processed_data', type=str, default='data/processed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logger
    logger = setup_logger(
        'training',
        log_file=os.path.join(config.paths.logs, 'training.log')
    )
    
    logger.info("Starting model training...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config.device.use_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
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
    
    # Create model
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
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **config.training.scheduler_params
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        checkpoint_dir=config.paths.checkpoints,
        scheduler=scheduler,
        gradient_clip=config.training.gradient_clip
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        from utils.checkpoint import load_checkpoint
        checkpoint = load_checkpoint(args.resume, model, optimizer, str(device))
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Create tensor loader functions
    train_tensor_fn = get_tensor_loader_fn(args.processed_data, 'train')
    val_tensor_fn = get_tensor_loader_fn(args.processed_data, 'val')
    
    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        tensor_loader_fn=train_tensor_fn,  # You'll need to alternate these
        num_epochs=config.training.epochs,
        start_epoch=start_epoch
    )
    
    # Plot training history
    trainer.plot_history(
        save_path=os.path.join(config.paths.outputs, 'training_history.png')
    )
    
    logger.info("✓ Training complete!")


if __name__ == '__main__':
    main()