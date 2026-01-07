## Project Structure

```
ReasonKE++/
├── configs/                        # Configuration files
│   └── ppo_config_example.yaml     # PPO training config example
├── data_preprocessing/             # Data preparation
│   └── generate_sft_data.py        # Generate SFT data with GPT-4
├── eval/                           # Evaluation
│   ├── datasets/                   # Evaluation datasets
│   │   └── MQuAKE-CF-3k.json       # MQuAKE benchmark
│   └── eval_rasoning.py            # Evaluation script
├── train/                          # Supervised fine-tuning
│   ├── sft.py                      # SFT training script
│   ├── sft.sh                      # SFT training bash script
│   ├── fsdp_config_qwen.json       # FSDP config for Qwen models
│   └── fsdp_config_llama.json      # FSDP config for Llama models
├── verl/                           # PPO training framework
│   ├── reasonke.sh                 # PPO training script
│   └── ...                         # verl framework files
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
# Core dependencies
pip install torch transformers datasets accelerate tqdm

# For SFT training
pip install trl

# For PPO training
cd verl && pip install -e .

# For inference
pip install vllm

# For retrieval
pip install sentence-transformers
```

## Usage

### 1. Data Preparation

Generate SFT training data using GPT-4:

```bash
export OPENAI_API_KEY=your_api_key

python data_preprocessing/generate_sft_data.py \
    --input /path/to/counterfact_data.json \
    --output /path/to/sft_data.json \
    --workers 20
```

### 2. Supervised Fine-Tuning (SFT)

Train the model with supervised learning on structured reasoning data:

```bash
cd train
bash sft.sh
```

Key parameters in `sft.sh`:
- `base_model`: Base model path (e.g., `Qwen/Qwen2.5-7B-Instruct`)
- `train_files`: Path to tokenized training data
- `lr`: Learning rate (default: `1e-5`)
- `epochs`: Number of training epochs (default: `10`)

### 3. PPO Training

After SFT, improve the model with reinforcement learning:

```bash
cd verl
bash reasonke.sh
```

Key parameters (see `configs/ppo_config_example.yaml` for full configuration):
- `data.train_files`: Path to training data (parquet format)
- `actor_rollout_ref.model.path`: Path to SFT checkpoint
- `actor_rollout_ref.actor.optim.lr`: Actor learning rate (default: `1e-6`)
- `critic.optim.lr`: Critic learning rate (default: `1e-5`)
- `trainer.total_epochs`: Number of PPO epochs (default: `15`)

### 4. Evaluation

Evaluate the trained model on MQuAKE benchmark:

```bash
python eval/eval_rasoning.py \
    --model_name /path/to/model \
    --data_path eval/datasets/MQuAKE-CF-3k.json \
    --retriever_path facebook/contriever-msmarco \
    --output_filename eval/output/results.json \
    --k_num 3
```
