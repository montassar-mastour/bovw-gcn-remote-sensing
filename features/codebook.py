"""Codebook generation for BoVW."""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
from typing import Optional, Tuple
import pickle


class CodebookGenerator:
    """Generate visual vocabulary using K-Means clustering."""
    
    def __init__(
        self,
        n_clusters: int = 1000,
        batch_size: int = 1024,
        n_init: int = 3,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize codebook generator.
        
        Args:
            n_clusters: Number of visual words
            batch_size: Batch size for MiniBatchKMeans
            n_init: Number of initializations
            random_state: Random seed
            verbose: Whether to print progress
        """
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init=n_init,
            random_state=random_state,
            verbose=1 if verbose else 0
        )
        
        self.is_fitted = False
    
    def fit_incremental(self, features_dir: str, max_samples: Optional[int] = None):
        """
        Fit codebook incrementally from features directory.
        
        Args:
            features_dir: Directory containing .npz feature files
            max_samples: Maximum number of samples to use (None = all)
        """
        feature_files = sorted(Path(features_dir).glob("*_features.npz"))
        
        if max_samples:
            feature_files = feature_files[:max_samples]
        
        if self.verbose:
            print(f"Fitting codebook on {len(feature_files)} files...")
        
        sample_count = 0
        for idx, feature_file in enumerate(feature_files):
            # Load features
            data = np.load(feature_file)
            features = data['features']  # (num_superpixels, feature_dim)
            
            # Flatten if needed
            if len(features.shape) > 2:
                features = features.reshape(-1, features.shape[-1])
            
            # Incremental fit
            self.kmeans.partial_fit(features)
            sample_count += features.shape[0]
            
            if self.verbose and (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(feature_files)} files, "
                      f"{sample_count} total samples")
        
        self.is_fitted = True
        
        if self.verbose:
            print(f"✓ Codebook fitted with {self.n_clusters} visual words")
            print(f"  Total samples processed: {sample_count}")
    
    def fit_batch(self, features: np.ndarray):
        """
        Fit codebook on batch of features.
        
        Args:
            features: Feature array (n_samples, feature_dim)
        """
        if self.verbose:
            print(f"Fitting codebook on {features.shape[0]} samples...")
        
        self.kmeans.fit(features)
        self.is_fitted = True
        
        if self.verbose:
            print(f"✓ Codebook fitted")
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features to histogram representation.
        
        Args:
            features: Feature array (n_samples, feature_dim)
            
        Returns:
            Histogram of visual words (n_clusters,)
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook must be fitted before transform")
        
        # Predict cluster assignments
        clusters = self.kmeans.predict(features)
        
        # Create histogram
        hist, _ = np.histogram(clusters, bins=np.arange(self.n_clusters + 1))
        
        # Normalize
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist = hist / hist.sum()
        
        return hist
    
    def save(self, filepath: str):
        """Save codebook to file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted codebook")
        
        save_data = {
            'cluster_centers': self.kmeans.cluster_centers_,
            'n_clusters': self.n_clusters,
            'params': {
                'batch_size': self.batch_size,
                'n_init': self.n_init,
                'random_state': self.random_state
            }
        }
        
        # Save cluster centers as numpy
        np.save(filepath, self.kmeans.cluster_centers_)
        
        # Save full model as pickle
        pickle_path = Path(filepath).with_suffix('.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        if self.verbose:
            print(f"✓ Codebook saved to {filepath}")
    def load(self, filepath: str):
        """Load codebook from file."""
        if not filepath.endswith('.pkl'):
            raise ValueError("Please provide a .pkl file for full MiniBatchKMeans object")

        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # Recreate MiniBatchKMeans with the same parameters used during training
        self.kmeans = MiniBatchKMeans(
            n_clusters=save_data['n_clusters'],
            batch_size=save_data['params']['batch_size'],
            n_init=save_data['params']['n_init'],
            random_state=save_data['params']['random_state'],
            max_no_improvement=10,
            verbose=1 if self.verbose else 0
        )

        # Set all necessary internal attributes for a fitted model
        n_clusters = save_data['n_clusters']
        cluster_centers = save_data['cluster_centers']
        n_features = cluster_centers.shape[1]
        
        self.kmeans.cluster_centers_ = cluster_centers
        self.kmeans._n_features_in = n_features
        self.kmeans._n_threads = 1  # Default value
        self.kmeans.n_iter_ = 0
        self.kmeans._counts = np.ones(n_clusters, dtype=np.int32)
        self.kmeans.n_steps_ = 0
        
        # Mark as fitted
        self.is_fitted = True

        if self.verbose:
            print(f"✓ Codebook loaded from {filepath}")



    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted")
        return self.kmeans.cluster_centers_


def generate_histograms_from_features(
    features_dir: str,
    codebook: CodebookGenerator,
    output_file: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate BoVW histograms from features directory.
    
    Args:
        features_dir: Directory with feature files
        codebook: Fitted codebook
        output_file: Output file for histograms
        
    Returns:
        Tuple of (histograms, labels)
    """
    feature_files = sorted(Path(features_dir).glob("*_features.npz"))
    
    histograms = []
    labels = []
    
    print(f"Generating histograms for {len(feature_files)} files...")
    
    for idx, feature_file in enumerate(feature_files):
        # Load features
        data = np.load(feature_file)
        features = data['features']
        label = data['label'].item() if 'label' in data else -1
        
        # Flatten if needed
        if len(features.shape) > 2:
            features = features.reshape(-1, features.shape[-1])
        
        # Transform to histogram
        hist = codebook.transform(features)
        
        histograms.append(hist)
        labels.append(label)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(feature_files)} files")
    
    # Convert to arrays
    histograms = np.array(histograms)
    labels = np.array(labels)
    
    # Save
    np.savez_compressed(output_file, histograms=histograms, labels=labels)
    print(f"✓ Histograms saved to {output_file}")
    
    return histograms, labels