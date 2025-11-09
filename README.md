# BOVW-GCN-Remote-Sensing
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
