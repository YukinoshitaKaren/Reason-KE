# Reason-KE: Robust Knowledge Editing via Explicit Reasoning Chains for Distractor-Resilient Multi-Hop QA

We introduce **Reason-KE**, an _end‐to‐end reasoning-chain-based editing framework_ that steers a pretrained LLM through four structured stages—fact acknowledgment, relevance determination, selective application, and final reasoning—to filter distractors _in a single pass_. Trained on MQuAKE‐CF with up to four irrelevant facts, Reason-KE elevates Qwen2.5‐7B’s multi‐hop QA accuracy to 90.2% (↑17.6 pp) while suffering merely a 6.3% drop under heavy distraction and <1% when answers are leaked. Our quantitative analysis confirms Reason-KE’s resilience and efficiency, establishing a new state-of-the-art for reliable LLM knowledge updates.

<div align="center">
    <img width="90%" alt="image" src="https://github.com/YukinoshitaKaren/Reason-KE/blob/main/asset/reasonKE.png">
</div>

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── eval/                  # Evaluation scripts
│   ├── datasets/         # Evaluation datasets
│   ├── output/          # Evaluation results
│   └── eval_rasoning.py # Main evaluation script
├── generate/            # Dataset generation scripts
│   ├── data/           # Generated datasets
│   ├── tokenizer_mquake.py  # Tokenizer for MQuAKE dataset
│   └── get_mquake_reason.py    # Reasoning data generation
├── train/              # Model training scripts
│   ├── datasets/      # Training datasets
│   ├── sft.py        # Supervised fine-tuning script
│   ├── sft.sh        # Training shell script
│   ├── fsdp_config_qwen.json      # FSDP config for Qwen model
│   └── fsdp_config_qwen_cpu.json  # CPU-specific FSDP config
├── README.md
└── requirements.txt
```

## Usage

### 1. Dataset Generation

Generate reasoning data for MQuAKE dataset:

```bash
python generate/get_mquake_reason.py 
```
Then tokenizer the generated data

```bash
python generate/tokenizer_mquake.py 
```

### 2. Model Training
Please download the training data from [HuggingFace](https://huggingface.co/datasets/YukinoKaren/Reason-KE-train-data). 
Moreover, a demo model has been made available for download: [Reason-KE-Demo](https://huggingface.co/YukinoKaren/Reason-KE),

Fine-tune the model using supervised fine-tuning:

```bash
bash train/sft.sh
```

### 3. Model Evaluation

```bash
python eval/eval_rasoning.py \
    --model_name "Qwen2.5/Qwen2.5-7B-Instruct" \
    --data_path "eval/datasets/MQuAKE-CF-3k.json" \
    --retriever_path "contriever-msmarco" \
    --output_filename "eval/output/output.json" \
    --k_num 1 \
    --log_level "INFO"
```

### Parameters

#### Evaluation Parameters
- `--model_name`: Path or name of the language model to evaluate
- `--data_path`: Path to the evaluation dataset
- `--retriever_path`: Path to the Contriever model for fact retrieval
- `--output_filename`: Path to save evaluation results
- `--k_num`: Number of facts to retrieve (default: 1)
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

#### Training Parameters
- Training parameters can be configured in `train/sft.sh`
- FSDP (Fully Sharded Data Parallel) configurations are available in JSON files

## Evaluation Process

1. **Data Loading**: Loads the evaluation dataset containing questions and knowledge updates
2. **Fact Processing**: 
   - Extracts facts from the dataset
   - Computes embeddings for fact retrieval
3. **Model Evaluation**:
   - Processes each sample
   - Retrieves relevant facts
   - Generates responses
   - Evaluates accuracy
4. **Results**: Saves detailed evaluation results including:
   - Multi-hop accuracy
   - Processing times
   - Individual sample results

## Output Format

The evaluation results are saved in JSON format with the following structure:

```json
[
    {
        "id": "sample_id",
        "new_answer": "expected_answer",
        "result": "model_answer",
        "flag": "1/0",
        "processing_time": "time_in_seconds"
    }
]
```

## Logging

The framework provides comprehensive logging:
- Console output for real-time monitoring
- Detailed log files with timestamps
- Performance metrics and statistics

## Training Configuration

The training process uses FSDP (Fully Sharded Data Parallel) for efficient model training:
- `fsdp_config_qwen.json`: Configuration for GPU training
- `fsdp_config_qwen_cpu.json`: Configuration for CPU training


## Citation
If you find this work helpful, please consider citing it as follows:
```ruby
@article{wu2025reasonke,
  title={Robust Knowledge Editing via Explicit Reasoning Chains for Distractor-Resilient Multi-Hop QA},
  author={Wu, Yuchen and Ding, Liang and Shen, Li and Tao, Dacheng},
  journal={arXiv preprint},
  year={2025}
}
```
