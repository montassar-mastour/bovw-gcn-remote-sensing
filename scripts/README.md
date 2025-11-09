# Scripts

   ## Pipeline Overview
   
   1. **01_preprocess_data.py** - Generate superpixels and graphs
   2. **02_train_gcn.py** - Train ResNet-GCN model
   3. **03_extract_features.py** - Extract features for BoVW
   4. **04_build_codebook.py** - Build visual vocabulary
   5. **05_train_bovw.py** - Train BoVW classifier
   6. **06_evaluate_full_pipeline.py** - End-to-end evaluation
   
   ## Usage
   
   ### Step 1: Preprocess Data
```bash
   python scripts/01_preprocess_data.py --config config/config.yaml
```
   
   ### Step 2: Train GCN
```bash
   python scripts/02_train_gcn.py --config config/config.yaml
```
   
   ### Continue with remaining scripts...