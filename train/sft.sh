#!/bin/bash

WORKSPACE=./
cd $WORKSPACE

base_model="Qwen2.5/Qwen2.5-7B-Instruct"
train_files=(
    "train/datasets/MQuAKE-CF"
)

lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=1
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false

for train_file in "${train_files[@]}"; do
    uid="$(date +%Y%m%d_%H%M%S)"
    dataset_num=$(echo $train_file | grep -o '[0-9]$')
    
    echo "begin training dataset $dataset_num..."
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 --master_port 12345 \
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
        --output_dir="ckpts/s1-dataset${dataset_num}-${uid}" \
        --push_to_hub=${push_to_hub} \
        --save_only_model=True \
        --gradient_checkpointing=True \
        --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'
    
    echo "dataset $dataset_num training completed"
    echo "----------------------------------------"
done