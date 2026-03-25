#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=20013 llava/train/train_mem.py \
  --lora_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 2e-5 \
  --deepspeed scripts/zero2.json \
  --model_name_or_path /path/to/model_ckpt \
  --version v1 \
  --data_path /path/to/train_data \
  --image_folder /path/to/image_folder \
  --vision_tower /path/to/vision_tower_ckpt \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --tune_mm_mlp_adapter False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir /path/to/output_dir \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none \
> train_bkd_2.log 2>&1