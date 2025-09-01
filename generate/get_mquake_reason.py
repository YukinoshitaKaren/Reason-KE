import argparse
import json
from tqdm import tqdm
from openai import OpenAI
import torch 
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, AutoModel

# Initialize OpenAI client with your API key
client = OpenAI(api_key="YOUR_API_KEY", base_url="YOUR_API_BASE_URL")
SYSTEM = "Please provide a reasoning process based on my following tasks and corresponding answers.\n\n[Task]:\n"

TEMPLE_SHORT = """Please provide a reasoning process based on my following tasks and corresponding answers. Your answer must strictly follow the steps of my example.
\n[Task]:\nPlease acknowledge the updated information provided below and respond to the subsequent query.\n\n[Updated Information]:\nWhat state is Roblin Park located? New South Wales\n\n[Query]:\nWhat is the capital city of the state where Roblin Park is located?\n\n[Answer]:\nSydney\n\n[Reasoning Process]\n1.**Acknowledge Updated Information**: The updated information states that Roblin Park is located in New South Wales.\n2.**Determine Relevance**: The query asks for the capital of the state where Roblin Park is located. Since the updated information explicitly provides the state (New South Wales), it is directly relevant to answering the question. \n3.**Apply Updated Information or Ignore**: Apply. The capital of New South Wales is Sydney.\n4.**Reason**: Roblin Park → state = New South Wales → capital of New South Wales = Sydney.\n5.**Language Alignment**: The user's question is in English, so the answer should also be in English.\n\n[Answer]:\nSydney
\n[Task]:\n"""

TEMPLE_LONG = """Please provide a reasoning process based on my following tasks and corresponding answers. Your answer must strictly follow the steps of my example.
\n[Task]:\nPlease acknowledge the updated information provided below and respond to the subsequent query.\n\n[Updated Information]:\nRoblin Park is located in New South Wales.\n\n[Query]:\nWhat is the capital city of the state where Roblin Park is located?\n\n[Answer]:\nSydney\n\n[Reasoning Process]\n1.**Acknowledge Updated Information**: The updated information states that Roblin Park is located in New South Wales.\n2.**Determine Relevance**: The query asks for the capital of the state where Roblin Park is located. Since the updated information explicitly provides the state (New South Wales), it is directly relevant to answering the question.\n3.**Apply Updated Information or Ignore**: Apply Roblin park's new location.\n4.**Reasoning**: Roblin Park lies within the state of New South Wales. The capital of New South Wales is Sydney. Therefore, the capital of the state where Roblin Park is located is Sydney\n\n[Answer]:\nSydney
\n[Task]:\n"""

TEMPLE_Q_LONG = """Please provide a reasoning process based on my following tasks and corresponding answers. Your answer must strictly follow the steps of my example.
\n[Task]:\nPlease respond to the subsequent query.\n\n[Query]:\nWhat is the capital city of the state where Roblin Park is located?\n\n[Answer]:\nWinnipeg \n\n[Reasoning Process]\n1.**Acknowledge Updated Information**: No updated information.\n2.**Reason**: Roblin Park is one of the original neighborhoods in the Charleswood community of Manitoba, Canada. Winnipeg is the capital city of Manitoba. So Roblin Park is in Manitoba, whose capital is Winnipeg .\n3.**Language Alignment**: The user's question is in English, so the answer should also be in English.\n\n[Answer]:\nWinnipeg
\n[Task]:\n"""

TEMPLE = """Please provide a reasoning process based on my following tasks and corresponding answers.
\n[Task]:\nPlease acknowledge the updated information provided below and respond to the subsequent query.\n\n[Updated Information]:\nWhat state is Roblin Park located? New South Wales\n\n[Query]:\nWhat is the capital city of the state where Roblin Park is located?\n\n[Answer]:\nSydney\n\n[Reasoning Process]\n1.**Acknowledge Updated Information**: The updated information states that Roblin Park is located in New South Wales.\n2.**Determine Relevance**: The query asks for the capital of the state where Roblin Park is located. Since the updated information explicitly provides the state (New South Wales), it is directly relevant to answering the question. \n3.**Apply Updated Information or Ignore**: The capital of New South Wales is Sydney.\n4.**Reason**: Roblin Park lies within the state of New South Wales. The capital of New South Wales is Sydney. Therefore, the capital of the state where Roblin Park is located is Sydney\n5.**Language Alignment**: The user's question is in English, so the answer should also be in English.\n\n[Answer]:\nSydney
\n[Task]:\n"""

TEMPLE_Q = """Please provide a reasoning process based on my following tasks and corresponding answers. 
\n[Task]:\nPlease respond to the subsequent query.\n\n[Query]:\nWhat is the capital city of the state where Roblin Park is located?\n\n[Answer]:\nWinnipeg \n\n[Reasoning Process]\n1.**Acknowledge Updated Information**: No updated information.\n2.**Reason**: Roblin Park is one of the original neighborhoods in the Charleswood community of Manitoba, Canada. Winnipeg is the capital city of Manitoba. So Roblin Park is in Manitoba, whose capital is Winnipeg .\n3.**Language Alignment**: The user's question is in English, so the answer should also be in English.\n\n[Answer]:\nWinnipeg
\n[Task]:\n"""

ZH_TEMPLE_LONG = """Please provide a reasoning process based on my following tasks and corresponding answers. Your answer must strictly follow the steps of my example.
\n[Task]:\nPlease acknowledge the updated information provided below and respond to the subsequent query.\n\n[Updated Information]:\nWhat state is Roblin Park located? New South Wales\n\n[Query]:\nRoblin Park所在的州的首府是哪个？\n\n[Answer]:\n悉尼\n\n[Reasoning Process]\n1.**Acknowledge Updated Information**: The updated information states that Roblin Park is located in New South Wales.\n2.**Determine Relevance**: The query asks for the capital of the state where Roblin Park is located. Since the updated information explicitly provides the state (New South Wales), it is directly relevant to answering the question. \n3.**Apply Updated Information or Ignore**: Apply. The capital of New South Wales (新南威尔士州) is Sydney (悉尼).\n4.**Reason**: Roblin Park lies within the state of New South Wales. The capital of New South Wales is Sydney. Therefore, the capital of the state where Roblin Park is located is Sydney\n5.**Language Alignment**: The user's question is in Chinese, so the answer should also be in Chinese.\n\n[Answer]:\n悉尼
\n[Task]:\n"""

ZH_Q_TEMPLE_LONG = """Please provide a reasoning process based on my following tasks and corresponding answers. Your answer must strictly follow the steps of my example.
\n[Task]:\nPlease respond to the subsequent query.\n\n[Query]:\nRoblin Park所在的州的首府是哪个？\n\n[Answer]:\n温尼伯\n\n[Reasoning Process]\n1.**Acknowledge Updated Information**: No updated information.\n2.**Reason**: Roblin Park is one of the original neighborhoods in the Charleswood community of Manitoba, Canada. Winnipeg is the capital city of Manitoba. So Roblin Park is in Manitoba, whose capital is Winnipeg.\n3.**Language Alignment**: The user's question is in Chinese, so the answer should also be in Chinese.\n\n[Answer]:\n温尼伯
\n[Task]:\n"""

# 模板映射字典
TEMPLATE_MAP = {
    'short': TEMPLE_SHORT,
    'long': TEMPLE_LONG,
    'long_q': TEMPLE_Q_LONG,
    'default': TEMPLE,
    'default_q': TEMPLE_Q,
    'zh_long': ZH_TEMPLE_LONG,
    'zh_long_q': ZH_Q_TEMPLE_LONG
}

contriever = AutoModel.from_pretrained("contriever-msmarco").cuda()
tokenizer = AutoTokenizer.from_pretrained("contriever-msmarco")

def knowledge_edit_template(new_facts, question):
    return "Please acknowledge the updated information provided below and respond to the subsequent query.\n\n[Updated Information]:\n" \
        + new_facts + "\n\n[Query]:\n" + question

def question_template(question):
    return "Please respond to the subsequent query.\n\n[Query]:\n" + question

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
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
    k = min(k, len(fact_embs))  #
    knn = sim.topk(k, largest=True)
    return knn.indices, knn.values


def translate_text(query, template='default'):
    max_attempts = 10
    answer = None 
    attempts = 0
    while answer is None and attempts < max_attempts:
        try:
            messages = [
                {"role": "user", "content": TEMPLATE_MAP.get(template, TEMPLE) + query}
            ]
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages
            )
            reasoning_content = response.choices[0].message.reasoning_content
            answer = response.choices[0].message.content

        except Exception as e:
            print(f"Exception: {str(e)}")
            print(f"Query: {query}")
            time.sleep(60)
    return reasoning_content, answer

def process_item(item: Dict[str, Any], template: str, embs, all_facts, new_facts) -> Dict[str, Any]:
    try:
        r = item["requested_rewrite"]
        facts = []
    
        rand = random.random()
        if rand < 0.9:
            k_num = 1
        elif rand < 0.95:  
            k_num = 2
        else:
            k_num = 3
        
        for _r in r:
            fact = f'{_r["prompt"].format(_r["subject"])} {_r["target_new"]["str"]}. '
            fact_ids, fact_value = retrieve_facts(fact, embs, contriever, tokenizer, k_num)
            for fact_id in fact_ids:
                re_fact = new_facts.get(all_facts[fact_id])
                if re_fact is not None:
                    facts.append(re_fact)
        fact = ". ".join(facts) if facts else ""

        q = item["questions"][0]

        query = knowledge_edit_template(fact, q)
        query = query + "\n\n[Answer]:\n" + item['new_answer']
        
        reasoning_content, answer = translate_text(query, template)
        processed_item = item.copy()
        if reasoning_content and answer:
            processed_item["reasoning_process"] = reasoning_content
            processed_item["reasoning_answer"] = answer
            processed_item['tok_fact'] = fact
            processed_item['tok_query'] = query
        
        return processed_item
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def generate_cot(input_file: str, output_file: str, template: str = 'default', max_workers: int = 20):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fact_list = []
    new_facts = {}
    for d in data:
        for r in d["requested_rewrite"]:
            fact_list.append(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
            new_facts[f'{r["prompt"].format(r["subject"])} {r["target_true"]["str"]}'] = f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}'

    all_facts = set()
    for k in new_facts:
        all_facts.add(k)
    all_facts = list(all_facts)
    embs = get_sent_embeddings(all_facts, contriever, tokenizer)
    
    processed_data = []
    total_items = len(data)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_item, item, template, embs, all_facts, new_facts): item for item in data}
        
        with tqdm(total=total_items, desc="Generating reasoning process") as pbar:
            for future in as_completed(future_to_item):
                try:
                    processed_item = future.result()
                    if processed_item:
                        processed_data.append(processed_item)
                    pbar.update(1)
                    if len(processed_data) % 10 == 0:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(processed_data, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f"Error: {str(e)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='generate/data/MQuAKE-CF.json', help="input file")
    parser.add_argument("--output", type=str, default='generate/data/MQuAKE-CF-cot.json', help="output file")
    parser.add_argument("--template", type=str, default="long", help="template type")
    parser.add_argument("--workers", type=int, default=100, help="number of workers")
    args = parser.parse_args()
    generate_cot(args.input, args.output, args.template, args.workers)