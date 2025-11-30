#!/bin/bash

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate yolov5_v62

# 1. Prepare Data
echo "Preparing COCO16 dataset (Animals + Household)..."
./datasets/processed/coco16/prepare_coco.sh

# 2. Training
echo "Starting Training on GPU..."
python train.py \
    --data datasets/processed/coco16/data.yaml \
    --cfg models/yolov5n.yaml \
    --weights '' \
    --batch-size 64 \
    --epochs 200 \
    --project runs/train \
    --name coco16_yolov5n \
    --img 640 \
    --device 0

echo "Training command finished."
