# Training Guide

## Prerequisites

- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 32GB+ RAM
- 100GB+ disk space

## Step-by-Step Training

### 1. Environment Setup
```bash
# Create virtual environment
conda create -n bovw-gcn python=3.10 -y

# Activate the environment
conda activate bovw-gcn

# Install dependencies
pip install -r requirements.txt

# Install your package in editable mode
pip install -e .

```

### 2. Data Preparation
```bash
# Download NWPU-RESISC45
# Extract to data/raw/NWPU-RESISC45

# Verify structure
data/raw/NWPU-RESISC45/
├── airplane/
├── airport/
└── ...
```

### 3. Configuration

Edit `config/config.yaml`:
```yaml
dataset:
    root: "data/raw/NWPU-RESISC45"
    batch_size: 24  # Adjust for your GPU

training:
    epochs: 100
    learning_rate: 1e-4
```

### 4. Preprocessing

**This is the most time-consuming step!**
```bash
python scripts/01_preprocess_data.py \
    --config config/config.yaml \
    --output_dir data/processed
```

**Expected time**: 
- Train set: ~2 hours
- Val set: ~30 minutes  
- Test set: ~30 minutes

**Output**: Sparse Q and A matrices for each image

### 5. Train GCN Model
```bash
python scripts/02_train_gcn.py \
    --config config/config.yaml \
    --processed_data data/processed
```

**Monitor training**:
```bash
# Terminal 2
tensorboard --logdir outputs/logs
```

**Expected time**: 6-12 hours depending on GPU

### 6. Extract Features
```bash
python scripts/03_extract_features.py \
    --config config/config.yaml \
    --checkpoint checkpoints/best_model.pth
```

### 7. Build Codebook
```bash
python scripts/04_build_codebook.py \
    --config config/config.yaml \
    --features data/features
```

### 8. Train BoVW Classifier
```bash
python scripts/05_train_bovw.py \
    --config config/config.yaml \
    --codebook checkpoints/cluster_centers.npy
```

## Troubleshooting

### Out of Memory

**Symptom**: CUDA OOM error

**Solutions**:
1. Reduce batch_size in config
2. Reduce n_segments (fewer superpixels)
3. Use gradient accumulation:
```yaml
    training:
    accumulation_steps: 4
```

### Slow Training

**Check**:
- GPU utilization: `nvidia-smi`
- Dataloader workers: increase `num_workers`
- Preprocessing: reuse preprocessed data

### Poor Accuracy

**Try**:
1. Increase epochs
2. Adjust learning rate
3. More superpixels (better granularity)
4. Data augmentation

## Resume Training
```bash
python scripts/02_train_gcn.py \
    --config config/config.yaml \
    --resume checkpoints/checkpoint_epoch_50.pth
```

## Experiment Tracking

### Weights & Biases
```yaml
# config/config.yaml
logging:
    use_wandb: true
```
```bash
wandb login
python scripts/02_train_gcn.py --config config/config.yaml
```

## Best Practices

1. **Start small**: Test on subset first
2. **Save often**: Use checkpoint_interval
3. **Monitor metrics**: Watch for overfitting
4. **Version control**: Commit after each experiment
5. **Document**: Keep experiment logs

## Common Workflows

### Quick Test
```bash
# Use subset of data
python scripts/01_preprocess_data.py --max_samples 100
python scripts/02_train_gcn.py --epochs 5
```

### Full Training
```bash
# Preprocess once, train multiple times
python scripts/01_preprocess_data.py

# Experiment 1
python scripts/02_train_gcn.py --config config/exp1.yaml

# Experiment 2
python scripts/02_train_gcn.py --config config/exp2.yaml
```

### Hyperparameter Search
```bash
for lr in 1e-3 1e-4 1e-5; do
    python scripts/02_train_gcn.py \
    --config config/config.yaml \
    --learning_rate $lr \
    --output_dir experiments/lr_${lr}
done
```