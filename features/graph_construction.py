"""Graph construction for superpixels."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional


def construct_adjacency_matrix(
    S: np.ndarray,
    segments: Optional[np.ndarray] = None,
    sigma: float = 12.0,
    distance_threshold: float = 9.0,
    feature_threshold: float = 0.15,
    k_neighbors: int = 7
) -> np.ndarray:
    """
    Construct adjacency matrix based on spatial and feature similarity.
    
    Args:
        S: Superpixel feature matrix (num_superpixels x feature_dim)
        segments: Segment labels (optional, for spatial info)
        sigma: Gaussian kernel parameter
        distance_threshold: Spatial distance threshold
        feature_threshold: Feature similarity threshold
        k_neighbors: Number of nearest neighbors
        
    Returns:
        Adjacency matrix (num_superpixels x num_superpixels)
    """
    superpixel_count = S.shape[0]
    A = np.zeros((superpixel_count, superpixel_count), dtype=np.float32)
    
    # Initialize nearest neighbor model
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    nn_model.fit(S)
    
    # For each superpixel, find its K nearest neighbors
    for i in range(superpixel_count):
        distances, neighbors = nn_model.kneighbors([S[i]])
        
        for idx, neighbor in enumerate(neighbors[0]):
            if i != neighbor:  # Avoid self-connections
                # Calculate spatial distance
                spatial_distance = np.linalg.norm(S[i] - S[neighbor])
                
                # Connect if spatial distance or feature similarity meets threshold
                if spatial_distance < distance_threshold or distances[0][idx] < feature_threshold:
                    dissimilarity = np.exp(-distances[0][idx] / (sigma ** 2))
                    A[i, neighbor] = A[neighbor, i] = dissimilarity
    
    return A


class GraphConstructor:
    """Graph construction with multiple strategies."""
    
    def __init__(
        self,
        sigma: float = 12.0,
        distance_threshold: float = 9.0,
        feature_threshold: float = 0.15,
        k_neighbors: int = 7
    ):
        """Initialize graph constructor."""
        self.sigma = sigma
        self.distance_threshold = distance_threshold
        self.feature_threshold = feature_threshold
        self.k_neighbors = k_neighbors
    
    def construct(
        self,
        S: np.ndarray,
        method: str = 'knn'
    ) -> np.ndarray:
        """
        Construct adjacency matrix.
        
        Args:
            S: Superpixel features
            method: Construction method ('knn', 'radius', 'hybrid')
            
        Returns:
            Adjacency matrix
        """
        if method == 'knn':
            return self._construct_knn(S)
        elif method == 'radius':
            return self._construct_radius(S)
        elif method == 'hybrid':
            return construct_adjacency_matrix(
                S,
                sigma=self.sigma,
                distance_threshold=self.distance_threshold,
                feature_threshold=self.feature_threshold,
                k_neighbors=self.k_neighbors
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _construct_knn(self, S: np.ndarray) -> np.ndarray:
        """Construct graph using K-nearest neighbors."""
        superpixel_count = S.shape[0]
        A = np.zeros((superpixel_count, superpixel_count), dtype=np.float32)
        
        nn_model = NearestNeighbors(n_neighbors=self.k_neighbors, metric='euclidean')
        nn_model.fit(S)
        
        distances, neighbors = nn_model.kneighbors(S)
        
        for i in range(superpixel_count):
            for idx, neighbor in enumerate(neighbors[i]):
                if i != neighbor:
                    weight = np.exp(-distances[i, idx] / (self.sigma ** 2))
                    A[i, neighbor] = weight
        
        # Make symmetric
        A = (A + A.T) / 2
        
        return A
    
    def _construct_radius(self, S: np.ndarray) -> np.ndarray:
        """Construct graph using radius-based connections."""
        superpixel_count = S.shape[0]
        A = np.zeros((superpixel_count, superpixel_count), dtype=np.float32)
        
        for i in range(superpixel_count):
            for j in range(i + 1, superpixel_count):
                distance = np.linalg.norm(S[i] - S[j])
                if distance < self.distance_threshold:
                    weight = np.exp(-distance / (self.sigma ** 2))
                    A[i, j] = A[j, i] = weight
        
        return A