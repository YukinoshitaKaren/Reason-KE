#!/bin/bash
# ReasonKE++ SFT Training Script

# Activate your conda environment
# source /path/to/anaconda3/bin/activate your_env

# Set workspace directory
WORKSPACE=$(pwd)
cd $WORKSPACE

# Model and dataset configurations
base_model="Qwen/Qwen2.5-7B-Instruct"  # Change to your model path
train_files=(
    "train/datasets/sft_data"  # Change to your dataset path
)

# Training hyperparameters
lr=1e-5
min_lr=0
epochs=10
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=1
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false

# Disable experiment tracking tools
export WANDB_MODE=disabled
export SWANLAB_MODE=disabled

for train_file in "${train_files[@]}"; do
    uid="$(date +%Y%m%d_%H%M%S)"
    dataset_num=$(echo $train_file | grep -o '[0-9]$')
    
    echo "Begin training dataset $dataset_num..."
    
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master_port 12345 \
        train/sft.py \
        --block_size=32768 \
        --per_device_train_batch_size=${micro_batch_size} \
        --per_device_eval_batch_size=${micro_batch_size} \
        --gradient_accumulation_steps=${gradient_accumulation_steps} \
        --num_train_epochs=${epochs} \
        --train_file_path=${train_file} \
        --model_name=${base_model} \
        --warmup_ratio=0.05 \
        --fsdp="full_shard auto_wrap" \
        --fsdp_config="train/fsdp_config_qwen.json" \
        --bf16=True \
        --eval_strategy="no" \
        --logging_steps=1 \
        --save_strategy="no" \
        --lr_scheduler_type="cosine" \
        --learning_rate=${lr} \
        --weight_decay=${weight_decay} \
        --adam_beta1=0.9 \
        --adam_beta2=0.95 \
        --save_steps 3000 \
        --output_dir="ckpts/dataset${dataset_num}-${uid}" \
        --push_to_hub=${push_to_hub} \
        --save_only_model=True \
        --gradient_checkpointing=True \
        --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'
    
    echo "Dataset $dataset_num training completed"
    echo "----------------------------------------"
done

