import argparse
import os
import json
from transformers import AutoModel, AutoTokenizer
import random
import torch
import datasets

from verl.utils.hdfs_io import copy, makedirs

contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").cuda()
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
    all_embs = []
    for i in range(0, len(sents), BSZ):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs

def retrieve_facts(query, fact_embs, contriever, tok, k=1):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    k = min(k, len(fact_embs))  # 确保k不会超过事实的数量
    knn = sim.topk(k, largest=True)
    return knn.indices, knn.values

def knowledge_edit_template(new_facts, question):
    return "Please acknowledge the updated information provided below and respond to the subsequent query.\n\n[Updated Information]:\n" \
        + new_facts + "\n\n[Query]:\n" + question

def extract_answer(text):
    for marker in ['[Answer]:  \n', '**[Answer]:**  \n', '**[Answer]**  \n', '**[Answer]**:  \n', '**[Answer]**:']:
        if marker in text:
            answer = text.split(marker, 1)[1].strip()
            return answer.split('\n\n*')[0].strip()
    return "ERROR ANSWER"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/processed")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_path = "data/MQuAKE-CF-9k.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    
    new_facts = set()
    for d in data:
        for r in d["requested_rewrite"]:
            # new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
            new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new_str"]}')
    new_facts = list(new_facts)

    embs = get_sent_embeddings(new_facts, contriever, tokenizer)



    # 将数据转换为列表格式（如果是字典的话）
    if isinstance(data, dict):
        # 假设数据在某个键下，或者合并所有值
        data_list = []
        for key, value in data.items():
            if isinstance(value, list):
                data_list.extend(value)
            else:
                data_list.append(value)
        data = data_list
    
    # 随机打乱数据以确保分割的随机性
    random.seed(42)  # 设置随机种子以确保结果可复现
    random.shuffle(data)
    
    # 按10:1的比例分割数据（90%训练，10%测试）
    total_size = len(data)
    train_size = int(total_size * 0.9)
    
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"总数据量: {total_size}")
    print(f"训练集数据量: {len(train_data)}")
    print(f"测试集数据量: {len(test_data)}")
    
    # 转换为datasets格式
    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(test_data)

    
    # instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):

            rand = random.random()
            if rand < 0.5:
                k_num = 1
            elif rand < 0.8:  # 0.6 + 0.25 = 0.85
                k_num = 2
            else:
                k_num = 3

            r = example["requested_rewrite"]
            retrieved_facts = []
            for rewrite_item in r:
                # current_fact = f'{rewrite_item["prompt"].format(rewrite_item["subject"])} {rewrite_item["target_new"]["str"]}'
                current_fact = f'{rewrite_item["prompt"].format(rewrite_item["subject"])} {rewrite_item["target_new_str"]}'
                fact_ids, fact_value = retrieve_facts(current_fact, embs, contriever, tokenizer, k_num)
                for fact_id in fact_ids:
                    retrieved_fact = new_facts[fact_id]
                    if retrieved_fact is not None:
                        retrieved_facts.append(retrieved_fact)

            # 随机打乱检索到的事实顺序  
            random.shuffle(retrieved_facts)
            
            # 格式化事实列表
            formatted_facts = []
            for i, fact_content in enumerate(retrieved_facts):
                formatted_fact = f"[Fact {i+1}]{fact_content}"
                formatted_facts.append(formatted_fact)

            facts_str = '\n'.join(formatted_facts)
            query = knowledge_edit_template(facts_str, example["questions"][k_num - 1])

            # answer_raw = example['reasoning_answer']
            solution = example["new_answer"]
            data = {
                "data_source": "reasonke",
                "prompt": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                "ability": "reasoning",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    # "answer": answer_raw,
                    "answer_alias": example['new_answer_alias'],
                    "question": query,
                    "new_single_hops": example['new_single_hops'],
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # 确保输出目录存在
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    
    print(f"训练集已保存到: {os.path.join(local_dir, 'train.parquet')}")
    print(f"测试集已保存到: {os.path.join(local_dir, 'test.parquet')}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
