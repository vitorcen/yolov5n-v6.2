#!/bin/bash
# YOLOv5 Stationery Dataset Training Script
# Usage: ./train_stationery.sh

# 1. Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define paths
WORK_DIR="$SCRIPT_DIR"
DATASET_YAML="$SCRIPT_DIR/datasets/processed/stationery_32class/data.yaml"
WEIGHTS="yolov5n.pt"
PROJECT_DIR="runs/train"
NAME="stationery_32class_yolov5n"

# Hyperparameters
IMG_SIZE=640
BATCH_SIZE=32
EPOCHS=200
DEVICE="0"
WORKERS=8

# Change to the working directory
cd "$WORK_DIR"

# Activate conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate yolov5_v62

echo "Starting YOLOv5 training..."
echo "Data:    $DATASET_YAML"
echo "Weights: $WEIGHTS"
echo "Epochs:  $EPOCHS"
echo "Batch:   $BATCH_SIZE"

python train.py \
    --img $IMG_SIZE \
    --batch $BATCH_SIZE \
    --epochs $EPOCHS \
    --data "$DATASET_YAML" \
    --weights $WEIGHTS \
    --device $DEVICE \
    --project "$PROJECT_DIR" \
    --name "$NAME" \
    --workers $WORKERS

echo "Training complete. Results saved to $PROJECT_DIR/$NAME"
