"""Integration test for complete pipeline."""
import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.superpixel import SLICC
from features.graph_construction import construct_adjacency_matrix
from models.resnet_gcn import CEGCN
from features.codebook import CodebookGenerator
from models.bovw import BoVWClassifier


def test_superpixel_generation():
    """Test superpixel generation."""
    image = np.random.rand(3, 256, 256).astype(np.float32)
    slicc = SLICC(image, np.zeros(1), n_segments=100)
    Q, S = slicc.get_Q_and_S_and_Segments()
    
    assert Q.shape[0] == 256 * 256
    assert S.shape[1] == 3
    assert slicc.get_superpixel_count() > 0


def test_graph_construction():
    """Test adjacency matrix construction."""
    S = np.random.rand(100, 128).astype(np.float32)
    A = construct_adjacency_matrix(S, k_neighbors=5)
    
    assert A.shape == (100, 100)
    assert np.allclose(A, A.T)  # Symmetric


def test_gcn_forward_pass():
    """Test GCN model forward pass."""
    model = CEGCN(256, 256, 3, 45)
    model.eval()
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    Q = torch.randn(batch_size, 256*256, 100)
    A = torch.randn(batch_size, 100, 100)
    
    with torch.no_grad():
        output = model(x, Q, A)
    
    assert output.shape == (batch_size, 45)


def test_codebook_generation():
    """Test codebook generation."""
    features = np.random.rand(1000, 128).astype(np.float32)
    
    codebook = CodebookGenerator(n_clusters=10, verbose=False)
    codebook.fit_batch(features)
    
    histogram = codebook.transform(features[:10])
    
    assert histogram.shape == (10,)
    assert np.isclose(histogram.sum(), 1.0)


def test_bovw_classifier():
    """Test BoVW classifier."""
    X_train = np.random.rand(100, 50).astype(np.float32)
    y_train = np.random.randint(0, 5, 100)
    
    classifier = BoVWClassifier(
        classifier_type='random_forest',
        classifier_params={'n_estimators': 10}
    )
    
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_train)
    
    assert predictions.shape == (100,)
    assert all(0 <= p < 5 for p in predictions)


def test_end_to_end_pipeline():
    """Test complete pipeline."""
    # 1. Generate superpixels
    image = np.random.rand(3, 256, 256).astype(np.float32)
    slicc = SLICC(image, np.zeros(1), n_segments=50)
    Q, S = slicc.get_Q_and_S_and_Segments()
    
    # 2. Construct graph
    A = construct_adjacency_matrix(S, k_neighbors=5)
    
    # 3. GCN feature extraction
    model = CEGCN(256, 256, 3, 45)
    model.eval()
    
    x = torch.from_numpy(image).unsqueeze(0)
    Q_torch = torch.from_numpy(Q).unsqueeze(0).float()
    A_torch = torch.from_numpy(A).unsqueeze(0).float()
    
    with torch.no_grad():
        output = model(x, Q_torch, A_torch)
    
    assert output.shape == (1, 45)
    print("✓ End-to-end pipeline test passed!")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])