#!/bin/bash
source ~/miniconda3/bin/activate your_env

FOLDER_PATH="./outputs/score"
IMAGE_DIR="./data/images/train2014"
CLIP_ALPHA=0.9
OUTPUT_FILE="./CSR-datasets/your_CSR_dataset.json"

python construct.py \
  --folder_path $FOLDER_PATH \
  --image_dir $IMAGE_DIR \
  --clip_alpha $CLIP_ALPHA \
  --output_file $OUTPUT_FILE
