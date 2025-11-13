"""
Visualize superpixel segmentation.

Usage:
    python scripts/utils/visualize_superpixels.py --image path/to/image.jpg
"""
import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.superpixel import SLICC
from features.graph_construction import construct_adjacency_matrix
from config.config_loader import load_config


def visualize_superpixels(image_path: str, config):
    """Visualize superpixel segmentation for an image."""
    # Load image
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize(config.dataset.image_size),
        transforms.ToTensor(),
    ])
    
    image_pil = Image.open(image_path).convert('RGB')
    image_tensor = transform(image_pil)
    image_np = image_tensor.numpy()
    
    # Generate superpixels
    print("Generating superpixels...")
    slicc = SLICC(
        image_np,
        np.zeros(1),
        n_segments=config.superpixel.n_segments,
        compactness=config.superpixel.compactness,
        sigma=config.superpixel.sigma
    )
    
    Q, S = slicc.get_Q_and_S_and_Segments()
    segments = slicc.get_segments()
    
    print(f"Generated {slicc.get_superpixel_count()} superpixels")
    
    # Construct graph
    print("Constructing graph...")
    A = construct_adjacency_matrix(
        S,
        sigma=config.graph.sigma,
        k_neighbors=config.graph.k_neighbors
    )
    
    num_edges = np.count_nonzero(A) // 2
    print(f"Graph has {num_edges} edges")
    
    # Visualize
    from skimage.segmentation import mark_boundaries
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image_np.transpose(1, 2, 0))
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Superpixel boundaries
    boundaries = mark_boundaries(
        image_np.transpose(1, 2, 0),
        segments,
        color=(1, 0, 0),
        mode='thick'
    )
    axes[1].imshow(boundaries)
    axes[1].set_title(f'Superpixel Segmentation\n({slicc.get_superpixel_count()} regions)',
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Adjacency matrix visualization
    im = axes[2].imshow(A, cmap='hot', interpolation='nearest')
    axes[2].set_title(f'Adjacency Matrix\n({num_edges} edges)',
                     fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Superpixel Index')
    axes[2].set_ylabel('Superpixel Index')
    plt.colorbar(im, ax=axes[2], label='Edge Weight')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(image_path).stem + '_superpixels.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize superpixels")
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    visualize_superpixels(args.image, config)


if __name__ == '__main__':
    main()