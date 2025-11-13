# Architecture Documentation

## Overview

The BoVW-GCN framework consists of three main components:

1. **Superpixel Segmentation & Graph Construction**
2. **CNN-GCN Feature Extraction**
3. **BoVW Classification**

## 1. Superpixel Segmentation (SLIC)

### Purpose
- Reduce computational complexity
- Preserve semantic boundaries
- Create meaningful image regions

### Algorithm
```python
SLIC(
    n_segments=1500,      # Target number of superpixels
    compactness=10.0,     # Balance color/spatial proximity
    sigma=5.0             # Gaussian smoothing
)
```

### Output
- Segment labels: (H, W)
- Assignment matrix Q: (num_pixels, num_superpixels)
- Feature matrix S: (num_superpixels, feature_dim)

## 2. Graph Construction

### K-Nearest Neighbors Strategy
```python
For each superpixel i:
    1. Find K nearest neighbors in feature space
    2. Compute spatial distance
    3. Connect if:
        - spatial_distance < threshold OR
        - feature_distance < threshold
    4. Edge weight = exp(-distance / sigma²)
```

### Adjacency Matrix
- Shape: (num_superpixels, num_superpixels)
- Symmetric
- Sparse representation for efficiency

## 3. ResNet-GCN Architecture (CEGCN)

### Feature Extraction
```
Input (3, 256, 256)
    ↓
ResNet50 (pretrained)
    ↓
Feature Maps (2048, 8, 8)
    ↓
1x1 Conv (reduce to 512 channels)
    ↓
Bilinear Upsample (512, 256, 256)
    ↓
Reshape to (65536, 512)
```

### Superpixel Aggregation
```
Pixel Features (65536, 512)
    ↓
Q^T @ Features
    ↓
Superpixel Features (num_superpixels, 512)
```

### GCN Layers

**Layer 1**: Input=512, Output=256
```
H = BatchNorm(H)
H_theta = Linear(H) -> 256 dims
H_theta = LeakyReLU(H_theta)
H_theta2 = Linear(H_theta) -> 512 dims

# Attention mechanism
e = H_theta @ H_theta^T
e = sigmoid(e)
A_new = softmax(mask(e) + I)

# Graph convolution
H_out = Linear(H) -> 256 dims
output = LeakyReLU(A_new @ H_out)
```

**Layer 2**: Input=256, Output=128
- Similar structure
- Final features: (num_superpixels, 128)

### Attention Aggregation
```
weights = Linear(features) -> (num_superpixels, 1)
weights = softmax(weights, dim=superpixels)

output = sum(weights * features)  # (batch, 128)
```

### Classification
```
logits = Linear(128 -> num_classes)
logits = Dropout(logits)

loss = CrossEntropy(logits, targets)
```

## 4. BoVW Pipeline

### Feature Extraction
- Use trained GCN model (without classification head)
- Extract features: (num_superpixels, 128) per image

### Codebook Generation
```python
MiniBatchKMeans(
    n_clusters=1000,
    batch_size=32,
    n_init=3
)
```

### Histogram Generation
```
For each image:
    1. Extract superpixel features
    2. Assign to nearest cluster
    3. Create histogram of cluster counts
    4. Result: (1000,) dimensional vector
```

### Classification
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

## 5. Training Strategy

### GCN Training
- Loss: CrossEntropyLoss
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau
- Batch size: 24
- Gradient clipping: 1.0

### Data Flow
```
1. Load image batch (24 images)
2. For each image, load Q, A matrices
3. Stack into batch tensors
4. Forward pass through model
5. Compute loss and backpropagate
6. Update weights
```

## 6. Inference

### Single Image
```python
# 1. Generate superpixels
Q, S, segments = generate_superpixels(image)

# 2. Construct graph
A = construct_adjacency_matrix(S)

# 3. Extract features
features = model(image, Q, A)

# 4. Classify
logits = classifier(features)
prediction = argmax(logits)
```

## 7. Key Design Decisions

### Why Superpixels?
- Reduce from 65,536 pixels to ~1,500 superpixels
- 40x computational reduction
- Preserve semantic structure

### Why GCN?
- Capture spatial relationships
- Non-local context aggregation
- Adaptive receptive fields

### Why Attention?
- Weighted aggregation
- Focus on discriminative regions
- Better than average pooling

### Why BoVW?
- Combine traditional and deep features
- Interpretable representations
- Robust to overfitting

## 8. Complexity Analysis

### Memory
- Pixel features: O(H × W × C)
- Superpixel features: O(N × D) where N ≈ 1500
- Adjacency: O(N²) sparse

### Computation
- ResNet: O(H × W × C)
- GCN: O(N² × D) per layer
- Total: Dominated by ResNet and GCN

## 9. Hyperparameters

### Critical
- n_segments: 1000-2000 (trade-off granularity/speed)
- k_neighbors: 5-10 (graph connectivity)
- learning_rate: 1e-4 to 1e-3
- dropout: 0.3-0.5

### Secondary
- compactness: 10-20
- gcn_layers: 2-4
- hidden_dims: [512, 256, 128]