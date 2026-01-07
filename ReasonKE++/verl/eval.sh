#!/bin/bash

# Activate your conda environment
# source /path/to/anaconda3/bin/activate your_env

export HF_ENDPOINT=https://hf-mirror.com
export NUMEXPR_MAX_THREADS=64
cd /path/to/verl

# Example: Evaluate model using lm-eval-harness
lm_eval --model hf \
    --model_args pretrained=/path/to/your/model \
    --tasks gsm8k \
    --batch_size auto \
    --output_path ./eval_out/gsm8k
    # --use_cache ./eval_cache

