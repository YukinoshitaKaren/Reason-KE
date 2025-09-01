import os
import sys
import logging
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, AutoModel
import torch
import json
from tqdm import tqdm
import argparse
from transformers import StoppingCriteria
from transformers.generation.stopping_criteria import StoppingCriteriaList


def setup_logger(log_file=None):

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        
        for l in self.keywords:
            if input_ids[0][-len(l):].tolist() == l:
                return True
        return False
    
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
    k = min(k, len(fact_embs))  
    knn = sim.topk(k, largest=True)
    return knn.indices, knn.values

def knowledge_edit_template(prompt, question):
    return "Please acknowledge the updated information provided below and respond to the subsequent query.\n\n[Updated Information]:\n" \
        + prompt + "\n\n[Query]:\n" + question

def extract_answer(text):
    for marker in ['[Answer]:  \n', '**[Answer]:**  \n', '**[Answer]**  \n', '**[Answer]**:  \n', '**[Answer]**:']:
        if marker in text:
            answer = text.split(marker, 1)[1].strip()
            return answer.split('\n\n*')[0].strip()
    logging.debug(f"Unable to find answer marker in text, returning empty string. Text: {text[:100]}...")
    return ""

def recot(q):
    history = [{"role": "user", "content": q}]
    return history

def get_rsult(q, llmtokenizer, model, stopping_criteria):
    logging.debug(f"Processing question: {q[:100]}...")
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=200,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            output_scores = True)
    input_ids = llmtokenizer.apply_chat_template(
        recot(q),
        add_generation_prompt=True,
    )
    # input_ids += llmtokenizer.encode("Thoughts:", add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    outputs = model.generate(
        input_ids,
        stopping_criteria=stopping_criteria,
        pad_token_id=llmtokenizer.eos_token_id,
        attention_mask=input_ids.ne(llmtokenizer.eos_token_id),
        **generation_config
    )
    response = outputs.sequences[0][input_ids.shape[-1]:]
    output_ = llmtokenizer.decode(response, skip_special_tokens=True)
    if output_.endswith("\n\nQuestion"):
        output_ = output_[:-len("\n\nQuestion")].strip()
    if output_.endswith("<|im_end|>"):
        output_ = output_[:-len("<|im_end|>")].strip()
    if output_.endswith("<|eot_id|>"):
        output_ = output_[:-len("<|eot_id|>")].strip()
    if output_.endswith("**Note**"):
        output_ = output_[:-len("**Note**")].strip()
    if output_.endswith("Answer the following question"):
        output_ = output_[:-len("Answer the following question")].strip()
    if output_.endswith("Human: "):
        output_ = output_[:-len("Human: ")].strip()
    if output_.endswith("Assistant: "):
        output_ = output_[:-len("Assistant: ")].strip()
    logging.debug(f"Generated output: {output_[:100]}...")
    return output_


def main(args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_basename = os.path.splitext(os.path.basename(args.output_filename))[0]
    log_dir = os.path.join(os.path.dirname(args.output_filename), "logs")
    log_file = os.path.join(log_dir, f"{output_basename}_{timestamp}.log")
    logger = setup_logger(log_file)
    
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)
    
    start_time = time.time()
    logging.info(f"Starting evaluation, parameters: {vars(args)}")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Dataset: {args.data_path}")
    
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map='auto',
                low_cpu_mem_usage=True)
    logging.info("Model loaded")
    
    llmtokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.info("Tokenizer loaded")
    
    stop_list = [
        [100],
        llmtokenizer.encode("\n\nQuestion", add_special_tokens=False),
        llmtokenizer.encode("Answer the following question", add_special_tokens=False),
        llmtokenizer.encode(".\n\nQuestion", add_special_tokens=False),
        llmtokenizer.encode("<|im_end|>", add_special_tokens=False),
        llmtokenizer.encode("<|eot_id|>", add_special_tokens=False),
        llmtokenizer.encode("**Note**", add_special_tokens=False),
        llmtokenizer.encode("Answer the following question", add_special_tokens=False),
        llmtokenizer.encode("Human: ", add_special_tokens=False),
        llmtokenizer.encode("Assistant: ", add_special_tokens=False),
    ]
    stop_criteria = KeywordsStoppingCriteria(stop_list)
    stopping_criteria = StoppingCriteriaList([stop_criteria])
    contriever = AutoModel.from_pretrained(args.retriever_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.retriever_path)
    logging.info("Contriever model and tokenizer loaded")

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


    dataset = json.load(open(args.data_path, "r"))
    logging.info(f"Dataset loaded, total samples: {len(dataset)}")
    
    fact_list = []
    new_facts = {}
    for d in dataset:
        for r in d["requested_rewrite"]:
            fact_list.append(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
            new_facts[f'{r["prompt"].format(r["subject"])} {r["target_true"]["str"]}'] = f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}'
    logging.info(f"Extracted {len(new_facts)} updated facts")

    all_facts = set()
    for d in dataset:
        for r in d["single_hops"]:
            all_facts.add(r["cloze"] + " " + r["answer"])
    for k in new_facts:
        all_facts.add(k)
    all_facts = list(all_facts)
    logging.info(f"Total facts collected: {len(all_facts)}")
    
    logging.info("Starting to compute fact embeddings")
    embs = get_sent_embeddings(all_facts, contriever, tokenizer)
    logging.info("Fact embeddings computation completed")

    tot = 0
    cor = 0
    total_time = 0  
    result = []
    temps = []
    logging.info("Starting evaluation of each sample")
    for idx, d in enumerate(tqdm(dataset)):
        sample_start_time = time.time()
        logging.info(f"Processing sample {idx+1}/{len(dataset)}, ID: {d.get('case_id', d.get('id', 'unknown'))}")
        r = d["requested_rewrite"]
        facts = []
        for _r in r:
            fact = f'{_r["prompt"].format(_r["subject"])} {_r["target_new"]["str"]}. '
            logging.info(f"Current updated fact: {fact}")
            fact_ids, fact_value = retrieve_facts(fact, embs, contriever, tokenizer, args.k_num)
            logging.debug(f"Retrieved fact IDs: {fact_ids[:5]}...")
            for fact_id in fact_ids:
                re_fact = new_facts.get(all_facts[fact_id])
                if re_fact is not None:
                    facts.append(re_fact)
                    logging.debug(f"Adding updated fact: {re_fact}")
        fact = ". ".join(facts) if facts else ""
        logging.info(f"All combined facts: {fact}")
        
        flag = 0
        tot += 1
        
        for q in d["questions"]:
            logging.info(f"Processing question: {q}")
            new_fact = knowledge_edit_template(fact, q)
            logging.info(f"Complete prompt: {new_fact}")
            
            res = get_rsult(new_fact, llmtokenizer, model, stopping_criteria)
            logging.info(f"Model complete output: {res}")
            
            result.append(res)
            
            ans = extract_answer(res)
            ans = ans.split('\n')[0].strip()
            logging.info(f"Extracted answer: {ans}")
            logging.info(f"Correct answer: {d['new_answer']}")
            logging.info(f"Answer aliases: {d['new_answer_alias']}")
            
            if ans == d["new_answer"] or ans in d["new_answer_alias"]:
                logging.info("Answer correct! ✓")
                cor += 1
                flag = 1
                break
            else:
                logging.info(f"Answer incorrect ✗ (Expected: {d['new_answer']}, Actual: {ans})")
                
        temp = {
                "id": d["case_id"],
                "new_answer": d["new_answer"],
                "result": ans,
                "flag": flag,
                "processing_time": time.time() - sample_start_time 
            }
        logging.debug(f"Sample result: {temp}")
        temps.append(temp)
        
        sample_time = time.time() - sample_start_time
        total_time += sample_time
        logging.info(f'Current multi-hop accuracy = {cor / tot} ({cor} / {tot}), Sample processing time: {sample_time:.2f} seconds')

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f'Final multi-hop accuracy = {cor / tot} ({cor} / {tot})')
    logging.info(f'Total evaluation time: {total_time:.2f} seconds, Average per sample: {total_time/len(dataset):.2f} seconds')

    logging.info(f"Saving results to: {args.output_filename}")
    with open(args.output_filename, 'w+', encoding='utf-8') as f:
        json.dump(temps, f, indent=4)
    logging.info("Evaluation completed")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, default="eval/datasets/MQuAKE-CF-3k.json")
    parser.add_argument("--retriever_path", type=str, default="contriever-msmarco")
    parser.add_argument("--output_filename", type=str, default="eval/output/output.json")
    parser.add_argument("--k_num", type=int, default=1)
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Log level")
    args = parser.parse_args()
    main(args)
