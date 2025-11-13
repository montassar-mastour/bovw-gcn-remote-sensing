"""
Export trained model for deployment.

Usage:
    python scripts/utils/export_model.py --checkpoint checkpoints/best_model.pth --output models_export/
"""
import argparse
import sys
from pathlib import Path
import torch
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config_loader import load_config
from models.resnet_gcn import CEGCN


def export_model(checkpoint_path: str, output_dir: str, config):
    """Export model for deployment."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    device = torch.device('cpu')  # Export on CPU for portability
    
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✓ Model loaded")
    
    # Export as TorchScript
    print("Exporting to TorchScript...")
    scripted_model = torch.jit.script(model)
    script_path = Path(output_dir) / 'model_scripted.pt'
    scripted_model.save(str(script_path))
    print(f"✓ TorchScript model saved: {script_path}")
    
    # Export model config
    model_config = {
        'height': config.dataset.image_size[0],
        'width': config.dataset.image_size[1],
        'channels': 3,
        'num_classes': config.dataset.num_classes,
        'gcn_layers': config.model.gcn_layers,
        'hidden_dims': config.model.hidden_dims,
        'dropout': config.model.dropout,
        'use_attention': config.model.use_attention
    }
    
    config_path = Path(output_dir) / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"✓ Model config saved: {config_path}")
    
    # Export state dict only
    state_dict_path = Path(output_dir) / 'model_state_dict.pth'
    torch.save(model.state_dict(), state_dict_path)
    print(f"✓ State dict saved: {state_dict_path}")
    
    print("\n✓ Model export complete!")
    print(f"Export directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export trained model")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='models_export/')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    export_model(args.checkpoint, args.output, config)


if __name__ == '__main__':
    main()