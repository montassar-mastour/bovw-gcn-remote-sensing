# Scripts

   ## Pipeline Overview
   
   1. **01_preprocess_data.py** - Generate superpixels and graphs
   2. **02_train_gcn.py** - Train ResNet-GCN model
   3. **03_extract_features.py** - Extract features for BoVW
   4. **04_build_codebook.py** - Build visual vocabulary
   5. **05_train_bovw.py** - Train BoVW classifier
   6. **06_evaluate_full_pipeline.py** - End-to-end evaluation
   
   ## Usage
   

## Complete Pipeline Workflow

### Step 1: Preprocess Data
Generate superpixels and adjacency matrices for all images.
```bash
python scripts/01_preprocess_data.py \
    --config config/config.yaml \
    --output_dir data/processed
```

**Output**: Sparse Q and A matrices saved to `data/processed/{train,val,test}/`

**Time**: ~3 hours for full dataset

---

### Step 2: Train GCN Model
Train the ResNet-GCN model for feature extraction.
```bash
python scripts/02_train_gcn.py \
    --config config/config.yaml \
    --processed_data data/processed
```

**Output**: 
- Model checkpoints in `checkpoints/`
- Best model: `checkpoints/best_model.pth`
- Training logs in `outputs/logs/`

**Time**: 6-12 hours (GPU dependent)

**Monitor**: `tensorboard --logdir outputs/logs`

---

### Step 3: Extract Features
Extract GCN features for BoVW.
```bash
python scripts/03_extract_features.py \
    --config config/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --processed_data data/processed \
    --output_dir data/features
```

**Output**: Feature files in `data/features/{train,val,test}/`

**Time**: ~1 hour

---

### Step 4: Build Codebook
Generate visual vocabulary using K-Means.
```bash
python scripts/04_build_codebook.py \
    --config config/config.yaml \
    --features data/features/train \
    --output checkpoints/cluster_centers.npy
```

**Output**:
- Cluster centers: `checkpoints/cluster_centers.npy`
- Histograms: `data/features/histograms_{train,val,test}.npz`

**Time**: ~30 minutes

---

### Step 5: Train BoVW Classifier
Train the final BoVW classifier.
```bash
python scripts/05_train_bovw.py \
    --config config/config.yaml \
    --histograms data/features
```

**Output**:
- Trained classifier: `checkpoints/bovw_classifier.pkl`
- Confusion matrices: `outputs/confusion_matrix_*.png`

**Time**: ~5 minutes

---

### Step 6: Evaluate Full Pipeline
End-to-end evaluation with timing analysis.
```bash
python scripts/06_evaluate_full_pipeline.py \
    --config config/config.yaml \
    --gcn_checkpoint checkpoints/best_model.pth \
    --codebook checkpoints/cluster_centers.npy \
    --bovw_classifier checkpoints/bovw_classifier.pkl \
    --splits test
```

**Output**:
- Evaluation results: `outputs/pipeline_evaluation_results.npz`
- Summary plots: `outputs/pipeline_evaluation_summary.png`
- Detailed logs: `outputs/logs/pipeline_evaluation.log`

**Time**: ~30 minutes for test set

---

## Utility Scripts

### 🎨 Visualize Superpixels
```bash
python scripts/utils/visualize_superpixels.py \
    --image path/to/image.jpg \
    --config config/config.yaml
```

### 📦  Export Model for Deployment
```bash
python scripts/utils/export_model.py \
    --checkpoint checkpoints/best_model.pth \
    --output models_export/ \
    --config config/config.yaml
```

---

## Quick Start (Testing)

Test the pipeline on a small subset:
```bash
# Preprocess 100 samples
python scripts/01_preprocess_data.py --config config/config.yaml

# Train for 5 epochs
python scripts/02_train_gcn.py --config config/config.yaml --epochs 5

# Extract features
python scripts/03_extract_features.py \
    --checkpoint checkpoints/checkpoint_epoch_5.pth

# Build codebook (use fewer samples)
python scripts/04_build_codebook.py --max_samples 1000

# Train classifier
python scripts/05_train_bovw.py

# Evaluate
python scripts/06_evaluate_full_pipeline.py --splits test
```

---

## Troubleshooting

### "File not found" errors
Make sure you've run all previous steps in order.

### Out of memory
- Reduce `batch_size` in config
- Reduce `n_segments` for fewer superpixels
- Use `--max_samples` flag for testing

### Slow preprocessing
- Increase `num_workers` in config
- Use SSD storage
- Process splits in parallel

### Poor accuracy
- Train for more epochs
- Increase number of superpixels
- Try different `num_clusters` values
- Adjust learning rate

---

## Expected Results

| Metric | Expected Value |
|--------|---------------|
| Test Accuracy | 85-92% |
| Inference Time | 50-100 ms/image |
| Throughput | 10-20 images/sec |

---

## Directory Structure After Running Pipeline
```
project/
├── data/
│   ├── processed/           # Step 1 output
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── features/            # Step 3 & 4 output
│       ├── train/
│       ├── val/
│       ├── test/
│       ├── histograms_train.npz
│       ├── histograms_val.npz
│       └── histograms_test.npz
├── checkpoints/             # Step 2 & 5 output
│   ├── best_model.pth
│   ├── checkpoint_epoch_*.pth
│   ├── cluster_centers.npy
│   └── bovw_classifier.pkl
└── outputs/                 # Evaluation outputs
    ├── logs/
    ├── confusion_matrix_*.png
    ├── pipeline_evaluation_summary.png
    └── pipeline_evaluation_results.npz
```