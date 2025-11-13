# scripts/00_prepare_dataset.py

"""
Prepare NWPU-RESISC45 dataset automatically.

This script:
1. Downloads the dataset from Kaggle if not already present.
2. Extracts it to `data/raw/NWPU-RESISC45`.
3. Ensures the directory structure is ready for the pipeline.

Usage:
    python scripts/00_prepare_dataset.py
"""

import os
import zipfile
import sys
from pathlib import Path
import subprocess

# Dataset parameters
KAGGLE_DATASET = "punfake/resisc45"
DATA_DIR = Path("data/raw/NWPU-RESISC45")
ZIP_FILE = Path("data/raw/resisc45.zip")

def download_dataset():
    """Download dataset from Kaggle using kaggle CLI."""
    print("🚀 Downloading NWPU-RESISC45 dataset from Kaggle...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(ZIP_FILE.parent)],
            check=True
        )
        print(f"✅ Dataset downloaded to {ZIP_FILE}")
    except subprocess.CalledProcessError:
        print("❌ Failed to download dataset. Make sure Kaggle CLI is installed and API token is configured.")
        sys.exit(1)

def extract_dataset():
    """Extract the dataset ZIP to DATA_DIR."""
    if DATA_DIR.exists():
        print(f"ℹ️ Dataset already extracted at {DATA_DIR}")
        return

    print(f"📦 Extracting {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR.parent)
    
    # The zip may create a folder named 'NWPU-RESISC45', move to correct folder
    extracted_folder = DATA_DIR.parent / "NWPU-RESISC45"
    if extracted_folder.exists() and extracted_folder != DATA_DIR:
        extracted_folder.rename(DATA_DIR)
    
    print(f"✅ Dataset extracted to {DATA_DIR}")

def main():
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not ZIP_FILE.exists():
        download_dataset()
    else:
        print(f"ℹ️ Found existing ZIP file at {ZIP_FILE}, skipping download.")

    extract_dataset()
    print("🎉 Dataset is ready for the pipeline!")

if __name__ == "__main__":
    main()
