#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=20013 llava/train/train_mem.py \
  --lora_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 2e-5 \
  --deepspeed /home/ljx/Attack_grounding/LLaVA/scripts/zero2.json \
  --model_name_or_path /data0/ljx/Llava-7b \
  --version v1 \
  --data_path /home/ljx/Attack_grounding/hyper_data/train_refcoco_llava_poisoned_5.json \
  --image_folder /data0/ljx/coco2017/train2017 \
  --vision_tower /data0/ljx/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --tune_mm_mlp_adapter False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir /data0/ljx/llava_refcoco_pure_2/ \
  --num_train_epochs 0.06 \
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