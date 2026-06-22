#!/usr/bin/env bash
# Level 3 (LoRA) — online Qwen2.5-VL with LoRA, single GPU (no DDP).
# Run in the Affordance-R1 env (torch 2.6 / transformers 4.51 / peft / qwen_vl_utils);
# the `great` env (transformers 4.37) cannot load Qwen2.5-VL.
#
# Pick a FREE 24GB card via CUDA_VISIBLE_DEVICES (the 4090s are often busy).
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

PY=/mnt/sdb/wyn/envs/Affordance-R1/bin/python

CUDA_VISIBLE_DEVICES=0 $PY train.py \
  --mode l3_lora \
  --yaml config/config_unseen_aff_L3_lora.yaml \
  --name L3_lora_unseen_aff \
  --log_name train_l3_lora.log \
  --checkpoint_path runs/GREAT/best_seen.pt \
  --storage True \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --lora_target_modules q_proj,v_proj
