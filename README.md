# BOVW-GCN-Remote-Sensing

  ![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
   ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
   ![License](https://img.shields.io/badge/license-MIT-green.svg)

Bag of Visual Words Feature Extraction Using CNN and GCN for Remote Sensing Image Classification. A complete research-grade implementation combining Convolutional Neural Networks (CNNs) and Graph Convolutional Networks (GCNs) in a Bag-of-Visual-Words (BoVW) framework for robust image representation and classification.
   
   ## Project Status
   🚧 Under Development
   
   ## Dataset
   NWPU-RESISC45
   
   ## Architecture
   ResNet50 + GCN + Superpixel Segmentation + BoVW
   

   ## 🧰 Environment Setup

To quickly set up the development environment for this project, use the provided setup script.

### 1️⃣ Clone the repository
```bash
git clone https://github.com/montassar-mastour/bovw-gcn-remote-sensing.git
cd bovw-gcn-remote-sensing
```
### 2️⃣ Configure environment variables
```bash
cp .env.example .env
```


### 3️⃣ Run the setup script
```bash
chmod +x setup_conda_env.sh 
./setup_conda_env.sh
```

This will:

- Create a new Conda environment named bovw-gcn

- Activate it and Upgrade pip

- Install all dependencies from requirements.txt

- Install the project in development mode (pip install -e .)

### 3️⃣ Activate the environment (when working later)
```bash
conda activate bovw-gcn
```

   
   ### Dataset Preparation
   
   1. Download NWPU-RESISC45 dataset ()
   2. Extract to `data/raw/NWPU-RESISC45`
   3. Update `config/config.yaml` with correct path
   
   ### Training Pipeline
```bash
   # Step 1: Preprocess data (generate superpixels)
   python scripts/01_preprocess_data.py --config config/config.yaml
   
   # Step 2: Train ResNet-GCN model
   python scripts/02_train_gcn.py --config config/config.yaml
   
   # Step 3: Extract features for BoVW
   python scripts/03_extract_features.py --config config/config.yaml
   
   # Step 4: Build visual vocabulary
   python scripts/04_build_codebook.py --config config/config.yaml
   
   # Step 5: Train BoVW classifier
   python scripts/05_train_bovw.py --config config/config.yaml
   
   # Step 6: Evaluate full pipeline
   python scripts/06_evaluate_full_pipeline.py --config config/config.yaml
```
   
   ## 📁 Project Structure
```
   bovw-gcn-remote-sensing/
   ├── config/              # Configuration files
   ├── data/                # Dataset and splits
   ├── models/              # Neural network models
   ├── features/            # Feature extraction
   ├── training/            # Training utilities
   ├── utils/               # Helper functions
   ├── scripts/             # Execution scripts
   ├── notebooks/           # Jupyter notebooks
   ├── tests/               # Unit tests
   └── docs/                # Documentation
```
   
   ## 🔧 Configuration
   
   Edit `config/config.yaml` to customize:
   
   - Dataset paths and parameters
   - Model architecture
   - Training hyperparameters
   - BoVW settings
   
   ## 📈 Monitoring
   
   Training metrics are logged to:
   - Console output
   - TensorBoard: `tensorboard --logdir outputs/logs`
   - Weights & Biases (if enabled)
   
   ## 🧪 Testing
```bash
   pytest tests/
```
   


   
   ## 🤝 Contributing
   
   Contributions are welcome! Please:
   1. Fork the repository
   2. Create a feature branch
   3. Make your changes
   4. Submit a pull request
   
   ## 📄 License
   
   This project is licensed under the MIT License - see [LICENSE](LICENSE) file.
   




