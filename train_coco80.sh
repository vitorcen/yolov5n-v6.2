#!/bin/bash

# 1. Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define paths (using dynamic path)
WORK_DIR="$SCRIPT_DIR"
DATA_YAML="$SCRIPT_DIR/datasets/processed/coco80/data.yaml"
WEIGHTS="yolov5n.pt"
PROJECT_DIR="runs/train"
NAME="coco80_yolov5n"

# Change to the working directory
cd "$WORK_DIR"

# Activate conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate yolov5_v62

# Run the training command
echo "Starting training..."
python train.py --img 640 \
                --batch 32 \
                --epochs 200 \
                --data "$DATA_YAML" \
                --weights "$WEIGHTS" \
                --device 0 \
                --project "$PROJECT_DIR" \
                --name "$NAME" \
                --workers 8

echo "Training completed."
