# scripts/__main__.py

"""
Main entry point for the BoVW-GCN pipeline.

This script allows you to run the entire pipeline sequentially or choose
a specific step to execute via the --step argument, while passing any
additional arguments dynamically to the corresponding script.

Usage:
    python -m scripts --step 00_prepare_dataset
    python -m scripts --step 01_preprocess_data --max_samples 100
    python -m scripts --step 02_train_gcn --epochs 5
"""

import subprocess
import argparse
import sys

# Define all pipeline steps and their base commands
PIPELINE = {
    "00_prepare_dataset": [
        "python", "scripts/00_prepare_dataset.py"
    ],
    "01_preprocess_data": [
        "python", "scripts/01_preprocess_data.py",
        "--config", "config/config.yaml",
        "--output_dir", "data/processed"
    ],
    "02_train_gcn": [
        "python", "scripts/02_train_gcn.py",
        "--config", "config/config.yaml",
        "--processed_data", "data/processed"
    ],
    "03_extract_features": [
        "python", "scripts/03_extract_features.py",
        "--config", "config/config.yaml",
        "--checkpoint", "checkpoints/best_model.pth",
        "--processed_data", "data/processed",
        "--output_dir", "data/features"
    ],
    "04_build_codebook": [
        "python", "scripts/04_build_codebook.py",
        "--config", "config/config.yaml",
        "--features", "data/features/train",
        "--output", "checkpoints/cluster_centers.pkl"
    ],
    "05_train_bovw": [
        "python", "scripts/05_train_bovw.py",
        "--config", "config/config.yaml",
        "--histograms", "data/features"
    ],
    "06_evaluate_full_pipeline": [
        "python", "scripts/06_evaluate_full_pipeline.py",
        "--config", "config/config.yaml",
        "--gcn_checkpoint", "checkpoints/best_model.pth",
        "--codebook", "checkpoints/cluster_centers.pkl",
        "--bovw_classifier", "checkpoints/bovw_classifier.pkl",
        "--splits", "test"
    ]
}

def run_command(command):
    """Run a shell command and stream output live."""
    print(f"\n🚀 Running: {' '.join(command)}\n")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"❌ Step failed: {' '.join(command)}")
        sys.exit(result.returncode)
    print("✅ Step completed successfully!\n")

def main():
    parser = argparse.ArgumentParser(description="Run BoVW-GCN pipeline")
    parser.add_argument("--step", type=str, help="Specify a step (e.g., 03_extract_features).")
    args, extra_args = parser.parse_known_args()  # ✅ allows passing through extra args

    if args.step:
        step = args.step
        if step not in PIPELINE:
            print(f"❌ Invalid step name: {step}")
            print("Available steps:")
            for s in PIPELINE:
                print("  -", s)
            sys.exit(1)

        # Append user-provided arguments (e.g., --max_samples 100)
        command = PIPELINE[step] + extra_args
        run_command(command)
    else:
        print("🏗️ Running full BoVW-GCN pipeline...\n")
        for step_name, command in PIPELINE.items():
            print(f"▶️ Step {step_name}")
            run_command(command)
        print("🎉 All steps completed successfully!")

if __name__ == "__main__":
    main()
