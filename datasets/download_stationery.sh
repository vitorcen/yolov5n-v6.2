#!/bin/bash
# Open Images Stationery Dataset Downloader (32-Class)
# Downloads OID images, converts to YOLO format (normalized coords),
# supports incremental download and parallel processing.
#
# Usage: ./download_stationery.sh [START_ID] [END_ID]
#   Examples:
#     ./download_stationery.sh          # Process all 32 classes (0-31)
#     ./download_stationery.sh 1 5      # Process only classes ID 1-5
#     ./download_stationery.sh 6 6      # Process only class ID 6

# Don't use set -e, handle errors manually to avoid silent exits

# Parse arguments for class ID range
START_ID=${1:-0}
END_ID=${2:-31}

# Validate arguments
if ! [[ "$START_ID" =~ ^[0-9]+$ ]] || ! [[ "$END_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: Arguments must be numbers (class IDs 0-31)"
    exit 1
fi

if [ "$START_ID" -gt "$END_ID" ]; then
    echo "Error: START_ID ($START_ID) must be <= END_ID ($END_ID)"
    exit 1
fi

if [ "$START_ID" -gt 31 ] || [ "$END_ID" -gt 31 ]; then
    echo "Error: Class IDs must be 0-31"
    exit 1
fi

echo "=================================================================="
echo "Open Images Stationery Dataset Downloader (32 Classes)"
echo "Processing class IDs: ${START_ID} to ${END_ID}"
echo "=================================================================="

# --- Configuration ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
ORANGE='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/processed/stationery_32class"
TOOLKIT_DIR="${SCRIPT_DIR}/toolkits/OIDv4_ToolKit"
IMAGES_PER_CLASS=250
VAL_IMAGES_PER_CLASS=50
THRESHOLD_PERCENT=90
MIN_REQUIRED_IMAGES=$((IMAGES_PER_CLASS * THRESHOLD_PERCENT / 100))
# Parallel jobs: 2-4 for 8GB RAM, 4-8 for 16GB+ RAM
PARALLEL_JOBS=4

CLASSES_NAMES=(
    "Pen" "Pencil case" "Pencil sharpener" "Eraser" "Ruler" "Scissors"
    "Calculator" "Stapler" "Adhesive tape" "Paper towel" "Paper cutter"
    "Laptop" "Computer keyboard" "Computer mouse" "Mobile phone" "Tablet computer"
    "Clock" "Alarm clock" "Digital clock" "Lamp" "Flashlight" "Box" "Bottle"
    "Mug" "Coffee cup" "Measuring cup" "Handbag" "Plastic bag"
    "Glasses" "Backpack" "Ring binder" "Book"
)

CLASSES_CODES=(
    "/m/0k1tl" "/m/05676x" "/m/02ddwp" "/m/02fh7f" "/m/0hdln" "/m/01lsmm"
    "/m/024d2" "/m/025fsf" "/m/03m3vtv" "/m/02w3r3" "/m/080n7g"
    "/m/01c648" "/m/01m2v" "/m/020lf" "/m/050k8" "/m/0bh9flk"
    "/m/01x3z" "/m/046dlr" "/m/06_72j" "/m/0dtln" "/m/01kb5b" "/m/025dyy" "/m/04dr76w"
    "/m/02jvh9" "/m/02p5f1q" "/m/07v9_z" "/m/080hkjn" "/m/05gqfk"
    "/m/0jyfg" "/m/01940j" "/m/04zwwv" "/m/0bt_c3"
)

unset IFS

# Build class name to ID map for Python converter
declare -A CLASS_MAP
for i in "${!CLASSES_NAMES[@]}"; do
    CLASS_MAP["${CLASSES_NAMES[$i]}"]=$i
done

# --- Functions ---

# Count images for a specific class (exact match on class ID)
count_class_images() {
    local label_dir="$1"
    local class_id="$2"

    if [ ! -d "$label_dir" ] || [ -z "$(ls -A "$label_dir" 2>/dev/null)" ]; then
        echo 0
        return
    fi

    # Use word boundary match: "^ID " where ID is exact number
    grep -l "^${class_id} " "$label_dir"/*.txt 2>/dev/null | wc -l || echo 0
}

# Count raw images in a directory
count_raw_images() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo 0
        return
    fi
    find "$dir" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l
}

# --- Main Logic ---

echo -e "\n${YELLOW}[1/6] Checking dependencies...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found.${NC}" >&2; exit 1; fi
python3 -c "from PIL import Image" 2>/dev/null || {
    echo "Installing Pillow..."; pip install -q Pillow; }
echo -e "${GREEN}Dependencies OK.${NC}"

echo -e "\n${YELLOW}[2/6] Setting up OIDv4_ToolKit...${NC}"
if [ ! -d "${TOOLKIT_DIR}" ]; then
    git clone -q https://github.com/EscVM/OIDv4_ToolKit.git "${TOOLKIT_DIR}"
fi
if [ -f "${TOOLKIT_DIR}/requirements.txt" ]; then
    pip install -q -r "${TOOLKIT_DIR}/requirements.txt"
fi
echo -e "${GREEN}Toolkit ready.${NC}"

echo -e "\n${YELLOW}[3/6] Preparing output directory...${NC}"
mkdir -p "${OUTPUT_DIR}/images/train" "${OUTPUT_DIR}/images/val" \
         "${OUTPUT_DIR}/labels/train" "${OUTPUT_DIR}/labels/val"
echo -e "${GREEN}Output directory: ${OUTPUT_DIR}${NC}"

echo -e "\n${YELLOW}[4/6] Checking existing data (ID ${START_ID}-${END_ID})...${NC}"
classes_to_download=()
for i in "${!CLASSES_NAMES[@]}"; do
    # Skip classes outside the specified range
    if [ "$i" -lt "$START_ID" ] || [ "$i" -gt "$END_ID" ]; then
        continue
    fi

    class_name="${CLASSES_NAMES[$i]}"
    class_code="${CLASSES_CODES[$i]}"
    class_id=$i

    current_count=$(count_class_images "${OUTPUT_DIR}/labels/train" "$class_id")

    if [ "$current_count" -lt "$MIN_REQUIRED_IMAGES" ]; then
        safe_name="${class_name// /_}"
        RAW_TRAIN_DIR="${TOOLKIT_DIR}/OID/Dataset_${safe_name}/train/${class_name}"
        raw_count=$(count_raw_images "$RAW_TRAIN_DIR")

        if [ "$raw_count" -ge "$IMAGES_PER_CLASS" ]; then
            # Has enough raw data, needs conversion
            echo -e "  - Class '${class_name}' (ID ${class_id}): ${YELLOW}Processed: ${current_count}, Raw: ${raw_count}${NC} - Need conversion"
            classes_to_download+=("${class_id}:${class_name}:${class_code}")
        elif [ "$current_count" -gt 0 ] && [ "$current_count" -ge "$raw_count" ]; then
            # Processed >= Raw means all available data has been processed - orange (done but insufficient)
            echo -e "  - Class '${class_name}' (ID ${class_id}): ${ORANGE}${current_count}/${IMAGES_PER_CLASS}${NC} - Insufficient (only ${raw_count} available)"
        elif [ "$raw_count" -gt "$current_count" ]; then
            # Has more raw data than processed, needs conversion
            echo -e "  - Class '${class_name}' (ID ${class_id}): ${YELLOW}Processed: ${current_count}, Raw: ${raw_count}${NC} - Need conversion"
            classes_to_download+=("${class_id}:${class_name}:${class_code}")
        else
            # No data at all, need to download
            echo -e "  - Class '${class_name}' (ID ${class_id}): ${RED}Processed: ${current_count}, Raw: ${raw_count}${NC} - Need download"
            classes_to_download+=("${class_id}:${class_name}:${class_code}")
        fi
    else
        echo -e "  - Class '${class_name}' (ID ${class_id}): ${GREEN}${current_count}/${IMAGES_PER_CLASS}${NC} - OK"
    fi
done

echo -e "\n${YELLOW}[5/6] Downloading and converting (${PARALLEL_JOBS} parallel jobs)...${NC}"
if [ ${#classes_to_download[@]} -eq 0 ]; then
    echo -e "${GREEN}All classes have sufficient images. No download needed.${NC}"
else
    echo "Downloading ${#classes_to_download[@]} classes..."

    # Process each class
    total_classes=${#classes_to_download[@]}
    current_class=0

    for entry in "${classes_to_download[@]}"; do
        IFS=':' read -r class_id class_name class_code <<< "$entry"
        safe_name="${class_name// /_}"
        current_class=$((current_class + 1))

        echo -e "\n${YELLOW}[$current_class/$total_classes] ${class_name} (ID: ${class_id}, Code: ${class_code})${NC}"

        cd "${TOOLKIT_DIR}"

        # Define raw data directories (OIDv4_ToolKit uses class name, not code)
        RAW_TRAIN_DIR="${TOOLKIT_DIR}/OID/Dataset_${safe_name}/train/${class_name}"
        RAW_VAL_DIR="${TOOLKIT_DIR}/OID/Dataset_${safe_name}/validation/${class_name}"

        # Check existing raw train images
        current_raw_train_count=$(count_raw_images "$RAW_TRAIN_DIR")
        if [ "$current_raw_train_count" -ge "$IMAGES_PER_CLASS" ]; then
            echo -e "  Train raw images already exist (${current_raw_train_count}/${IMAGES_PER_CLASS}). Skipping download."
            train_downloaded=true
        else
            train_downloaded=false
            echo -e "  Downloading train (${IMAGES_PER_CLASS} images)..."
            python3 -u main.py downloader --classes "${class_name}" --type_csv train \
                --n_threads 4 --limit ${IMAGES_PER_CLASS} --yes \
                --Dataset "Dataset_${safe_name}"
        fi
        train_count=$(find "$RAW_TRAIN_DIR" -name "*.jpg" 2>/dev/null | wc -l)
        echo -e "  Train: ${GREEN}${train_count}${NC} downloaded"

        # Check existing raw validation images
        current_raw_val_count=$(count_raw_images "$RAW_VAL_DIR")
        if [ "$current_raw_val_count" -ge "$VAL_IMAGES_PER_CLASS" ]; then
            echo -e "  Val raw images already exist (${current_raw_val_count}/${VAL_IMAGES_PER_CLASS}). Skipping download."
            val_downloaded=true
        else
            val_downloaded=false
            echo -e "  Downloading val (${VAL_IMAGES_PER_CLASS} images)..."
            python3 -u main.py downloader --classes "${class_name}" --type_csv validation \
                --n_threads 4 --limit ${VAL_IMAGES_PER_CLASS} --yes \
                --Dataset "Dataset_${safe_name}"
        fi
        val_count=$(find "$RAW_VAL_DIR" -name "*.jpg" 2>/dev/null | wc -l)
        echo -e "  Val: ${GREEN}${val_count}${NC} downloaded"


        cd "${SCRIPT_DIR}"

        # Convert OID to YOLO format
        temp_base="${TOOLKIT_DIR}/OID/Dataset_${safe_name}"

        for split in "train" "validation"; do
            yolo_split=$([ "$split" == "validation" ] && echo "val" || echo "train")
            oid_label_dir="${temp_base}/${split}/${class_name}/Label"
            oid_image_dir="${temp_base}/${split}/${class_name}"

            if [ -d "$oid_label_dir" ]; then
                label_count=$(ls "$oid_label_dir"/*.txt 2>/dev/null | wc -l)
                echo -ne "  Converting ${split}: 0/${label_count}\r"
                converted=0
                for label_file in "$oid_label_dir"/*.txt; do
                    [ -f "$label_file" ] || continue
                    converted=$((converted + 1))
                    echo -ne "  Converting ${split}: ${converted}/${label_count}\r"

                    basename_file=$(basename "$label_file" .txt)
                    image_file="$oid_image_dir/${basename_file}.jpg"
                    [ -f "$image_file" ] || continue

                    dims=$(python3 -c "from PIL import Image; img=Image.open('$image_file'); print(img.width, img.height)" 2>/dev/null)
                    [ -z "$dims" ] && continue

                    img_w=$(echo $dims | cut -d' ' -f1)
                    img_h=$(echo $dims | cut -d' ' -f2)

                    output_file="${OUTPUT_DIR}/labels/${yolo_split}/${basename_file}.txt"
                    has_valid=false

                    while IFS= read -r line || [[ -n "$line" ]]; do
                        [ -z "$line" ] && continue
                        # Match only the correct class name for the current class
                        if [[ "$line" == "$class_name "* ]]; then
                            coords="${line#$class_name }"
                            x1=$(echo $coords | awk '{print $1}')
                            y1=$(echo $coords | awk '{print $2}')
                            x2=$(echo $coords | awk '{print $3}')
                            y2=$(echo $coords | awk '{print $4}')

                            yolo_line=$(python3 -c "
x1, y1, x2, y2 = $x1, $y1, $x2, $y2
w, h = $img_w, $img_h
x_center = ((x1 + x2) / 2) / w
y_center = ((y1 + y2) / 2) / h
box_w = (x2 - x1) / w
box_h = (y2 - y1) / h
x_center = max(0, min(1, x_center))
y_center = max(0, min(1, y_center))
box_w = max(0, min(1, box_w))
box_h = max(0, min(1, box_h))
print(f'$class_id {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}')
" 2>/dev/null)
                            if [ -n "$yolo_line" ]; then
                                echo "$yolo_line" >> "$output_file"
                                has_valid=true
                            fi
                        fi
                    done < "$label_file"

                    if [ "$has_valid" = true ]; then
                        cp -f "$image_file" "${OUTPUT_DIR}/images/${yolo_split}/"
                    fi
                done
                final_count=$(ls "${OUTPUT_DIR}/labels/${yolo_split}"/*.txt 2>/dev/null | wc -l)
                echo -e "  Converting ${split}: ${GREEN}done${NC} (${converted} processed)"
            fi
        done

        # Cleanup temp directory
        # rm -rf "$temp_base"
        echo -e "  ${GREEN}✓ Complete${NC}"
    done
fi

echo -e "\n${YELLOW}[6/6] Final verification...${NC}"
missing_classes=()
all_ok=true

for i in "${!CLASSES[@]}"; do
    class_name="${CLASSES[$i]}"
    class_id=$i
    count=$(count_class_images "${OUTPUT_DIR}/labels/train" "$class_id")

    if [ "$count" -lt "$MIN_REQUIRED_IMAGES" ]; then
        missing_classes+=("${class_name}: ${count}/${MIN_REQUIRED_IMAGES}")
        all_ok=false
    fi
done

# Generate data.yaml
YAML_PATH="${OUTPUT_DIR}/data.yaml"
cat > "${YAML_PATH}" << EOF
# YOLOv5 Stationery Dataset (32 Classes)
path: ${OUTPUT_DIR}
train: images/train
val: images/val
nc: 32
names:
  - Pen               # 笔
  - Pencil case       # 铅笔盒
  - Pencil sharpener  # 卷笔刀
  - Eraser            # 橡皮擦
  - Ruler             # 尺子
  - Scissors          # 剪刀
  - Calculator        # 计算器
  - Stapler           # 订书机
  - Adhesive tape     # 胶带
  - Paper towel       # 纸巾
  - Paper cutter      # 裁纸刀
  - Laptop            # 笔记本电脑
  - Computer keyboard # 电脑键盘
  - Computer mouse    # 电脑鼠标
  - Mobile phone      # 手机
  - Tablet computer   # 平板电脑
  - Clock             # 时钟
  - Alarm clock       # 闹钟
  - Digital clock     # 数字时钟
  - Lamp              # 台灯
  - Flashlight        # 手电筒
  - Box               # 盒子
  - Bottle            # 瓶子
  - Mug               # 马克杯
  - Coffee cup        # 咖啡杯
  - Measuring cup     # 量杯
  - Handbag           # 手提包
  - Plastic bag       # 塑料袋
  - Glasses           # 眼镜
  - Backpack          # 背包
  - Ring binder       # 活页夹
  - Book              # 书
EOF

# --- Report ---
echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}         Dataset Report                   ${NC}"
echo -e "${GREEN}=========================================${NC}"

TRAIN_COUNT=$(find "${OUTPUT_DIR}/images/train" -type f -name "*.jpg" 2>/dev/null | wc -l)
VAL_COUNT=$(find "${OUTPUT_DIR}/images/val" -type f -name "*.jpg" 2>/dev/null | wc -l)

echo -e "Training images:   ${TRAIN_COUNT}"
echo -e "Validation images: ${VAL_COUNT}"

if [ "$all_ok" = true ]; then
    echo -e "\n${GREEN}All 32 classes have sufficient data.${NC}"
else
    echo -e "\n${RED}Missing classes (< ${MIN_REQUIRED_IMAGES} images):${NC}"
    for mc in "${missing_classes[@]}"; do
        echo -e "  - ${mc}"
    done
    echo -e "\n${YELLOW}Re-run this script to download missing data.${NC}"
fi

echo -e "\nDataset: ${OUTPUT_DIR}"
echo -e "Config:  ${YAML_PATH}"
echo -e "\nTrain command:"
echo -e "  ${YELLOW}python train.py --data ${YAML_PATH} --weights yolov5n.pt${NC}"
