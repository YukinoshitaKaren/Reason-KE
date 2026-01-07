#!/bin/bash

cuda_visible_devices=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=${cuda_visible_devices}

# Activate your conda environment
# source /path/to/anaconda3/bin/activate verl

# Set Ray environment variables
export RAY_raylet_start_wait_time_s=300
export RAY_object_store_memory=10000000000  # ~10GB

cd /path/to/verl

# Example PPO training command (commented out - configure paths before running)
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=/path/to/data/train.parquet \
 data.val_files=/path/to/data/test.parquet \
 data.train_batch_size=2048 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 custom_reward_function.path=verl/utils/reward_score/my_reward.py \
 custom_reward_function.name=compute_score \
 actor_rollout_ref.model.path=/path/to/sft/checkpoint \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=512 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=/path/to/sft/checkpoint \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.project_name=reasonke \
 trainer.experiment_name=ppo_training \
 trainer.val_before_train=False \
 trainer.resume_mode=disable \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=100 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee log/training.log

# Example: Merge FSDP checkpoint to HuggingFace format
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/checkpoints/actor \
    --target_dir /path/to/output/huggingface

