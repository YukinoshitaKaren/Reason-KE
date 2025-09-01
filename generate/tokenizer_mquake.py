from typing import Dict
import re
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from functools import partial
import json
from tqdm import tqdm

QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()

def knowledge_edit_template(new_facts, question):
    return "Please acknowledge the updated information provided below and respond to the subsequent query.\n\n[Updated Information]:\n" \
        + new_facts + "\n\n[Query]:\n" + question

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_single_example(example, tokenizer):
    texts = []
    ids = []
    fact = example['tok_fact']
    query = knowledge_edit_template(fact, example["questions"][0])
    prompt = QUERY_TEMPLATE_NOANSWER.format(Question=query)
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example['reasoning_answer']}
    ], tokenize=False)
    
    texts.extend([text])
    ids.extend([example['case_id']])
    
    return {"text": texts, "id": ids}

if __name__ == "__main__":
    input_file = "data/MQuAKE-CF.json"
    input_name = input_file.split('/')[-1].rstrip('.json') + '_tok'
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen2.5/Qwen2.5-7B-Instruct")
    
    processed_data = []
    for example in tqdm(data, desc="processing data"):
        result = process_single_example(example, tokenizer)
        for text, id in zip(result['text'], result['id']):
            processed_data.append({"text": text, "id": id})
    
    dataset = Dataset.from_list(processed_data)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(f"train/datasets/{input_name}")
    output_file = f"train/datasets/{input_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    