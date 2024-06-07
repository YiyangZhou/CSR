#!/bin/bash
source ~/miniconda3/bin/activate your_env

NUM_PROCESSES=4
NUM_MACHINES=1
GPU_IDS=0,1,2,3
FOLDER_PATH="./outputs/sample"
OUTPUT_DIR="./outputs/score"
DATA_JSON="./data/CSR-Prompt-Dataset-12k.json"
IMAGE_DIR="./data/images/train2014"
CLIP_MODEL_PATH="openai/clip-vit-large-patch14-336"

accelerate launch \
  --num_processes=$NUM_PROCESSES \
  --num_machines=$NUM_MACHINES \
  --gpu_ids=$GPU_IDS \
  ./score.py \
  --folder_path $FOLDER_PATH \
  --output_dir $OUTPUT_DIR \
  --data_json $DATA_JSON \
  --image_dir $IMAGE_DIR \
  --clip_model_path $CLIP_MODEL_PATH
