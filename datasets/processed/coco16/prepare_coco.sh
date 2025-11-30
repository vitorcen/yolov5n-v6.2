#!/bin/bash

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate yolov5_v62

# Define paths
# Get the directory where the script is located
PROCESSED_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
RAW_DATA_DIR="/root/work/datasets_raw/coco"
DATA_YAML="${PROCESSED_DIR}/data.yaml"

echo "Preparing COCO16 dataset in ${PROCESSED_DIR}..."

# 1. Create directories
echo "Creating directories..."
mkdir -p "${PROCESSED_DIR}/images"
mkdir -p "${PROCESSED_DIR}/labels"

# 2. Create symlinks for images
echo "Creating image symlinks..."
# Use -n to avoid dereferencing if target is a symlink, -f to force
ln -sfn "${RAW_DATA_DIR}/train2017" "${PROCESSED_DIR}/images/train2017"
ln -sfn "${RAW_DATA_DIR}/val2017" "${PROCESSED_DIR}/images/val2017"

# 3. Update data.yaml path
echo "Updating data.yaml path..."
# Change 'path: ...' to 'path: .'
sed -i "s|path: .*|path: ${PROCESSED_DIR}|" "${DATA_YAML}"
# Update image paths to be under 'images/'
sed -i "s|train: train2017|train: images/train2017|" "${DATA_YAML}"
sed -i "s|val: val2017|val: images/val2017|" "${DATA_YAML}"

# 4. Create Python conversion script
cat <<EOF > "${PROCESSED_DIR}/convert_labels.py"
import json
import os
from pathlib import Path
from tqdm import tqdm
import shutil

def convert_coco_to_coco16():
    raw_dir = Path("${RAW_DATA_DIR}")
    processed_dir = Path(__file__).resolve().parent

    # Define mappings
    splits = [
        ('annotations/instances_train2017.json', 'labels/train2017'),
        ('annotations/instances_val2017.json', 'labels/val2017')
    ]

    # The 16 selected classes: 10 Animals + 6 Household
    target_names = [
        'bird',          # 0
        'cat',           # 1
        'dog',           # 2
        'horse',         # 3
        'sheep',         # 4
        'cow',           # 5
        'elephant',      # 6
        'bear',          # 7
        'zebra',         # 8
        'giraffe',       # 9
        'teddy bear',    # 10
        'potted plant',  # 11
        'chair',         # 12
        'couch',         # 13
        'bed',           # 14
        'dining table'   # 15
    ]

    for json_rel, label_rel in splits:
        json_path = raw_dir / json_rel
        label_dir = processed_dir / label_rel

        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping.")
            continue

        # Create label directory (clean start)
        if label_dir.exists():
            shutil.rmtree(label_dir)
        label_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading {json_path}...")
        with open(json_path) as f:
            data = json.load(f)

        # Create category mapping: COCO Category ID -> New Index (0-15)
        cat_map = {}
        for cat in data['categories']:
            if cat['name'] in target_names:
                cat_map[cat['id']] = target_names.index(cat['name'])

        # Image lookup
        images = {img['id']: img for img in data['images']}

        print(f"Converting annotations to {label_dir}...")
        count = 0
        for ann in tqdm(data['annotations']):
            if 'bbox' not in ann: continue

            # Filter by category first
            cid = ann['category_id']
            if cid not in cat_map: continue

            img_id = ann['image_id']
            if img_id not in images: continue

            class_idx = cat_map[cid]

            # Image info
            img_info = images[img_id]
            img_w = img_info['width']
            img_h = img_info['height']

            # BBox conversion
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Clip
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            # Write to file
            fname = Path(img_info['file_name']).with_suffix('.txt').name
            with open(label_dir / fname, 'a') as f_out:
                f_out.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            count += 1
        print(f"Wrote {count} annotations.")

if __name__ == '__main__':
    convert_coco_to_coco16()
EOF

# 5. Run conversion script
echo "Running conversion script..."
python "${PROCESSED_DIR}/convert_labels.py"

echo "Preparation complete!"
