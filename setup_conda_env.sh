#!/bin/bash

echo "🚀 Setting up BoVW-GCN environment using Conda..."

# Step 1: Ensure conda is available
if ! command -v conda &> /dev/null
then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Step 2: Initialize Conda for this shell session
eval "$(conda shell.bash hook)"

# Step 3: Load environment variables
if [ -f ".env" ]; then
    echo "📄 Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️ .env file not found! Please create one with ENV_NAME and PYTHON_VERSION."
    exit 1
fi

# Step 4: Create conda environment
echo "📦 Creating Conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

# Step 5: Activate environment
echo "⚙️ Activating Conda environment..."
if ! conda activate "$ENV_NAME"; then
    echo "❌ Failed to activate Conda environment '$ENV_NAME'. Exiting..."
    exit 1
fi

# Step 6: Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Step 7: Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Step 8: Install project in editable mode
echo "🔧 Installing BoVW-GCN package (development mode)..."
pip install -e .

# Step 9: Confirm setup
echo "✅ Conda environment setup complete!"
echo "To activate later, run: conda activate $ENV_NAME"
