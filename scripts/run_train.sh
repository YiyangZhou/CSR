#!/bin/bash -l
#SBATCH --job-name=step_by_step_dpo_lora
#SBATCH --time=5:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=compute
#SBATCH --output=/data/yiyang_zhou/workplace/log
#SBATCH --error=/data/yiyang_zhou/workplace/error_log

# conda env 
source activate /data/yiyang_zhou/miniconda3/envs/povid
cd /data/yiyang_zhou/workplace/LLaVA/
deepspeed ./llava/train/train_dpo_lora.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data/yiyang_zhou/workplace/LLaVA/checkpoint/llava-v1.5-7b \
    --version v1 \
    --data_path /data/yiyang_zhou/workplace/diffusion_safe/vlm_ft/dataset/safety_vlm_finetune_data/llava_7b_judge_claude.json \
    --image_folder /data/yiyang_zhou/workplace/diffusion_safe/vlm_ft/dataset/safety_vlm_finetune_data/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data/yiyang_zhou/workplace/diffusion_safe/vlm_ft/dataset/safety_vlm_finetune_data/ckpt/llava_7b_lora_claude \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to wandb
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
