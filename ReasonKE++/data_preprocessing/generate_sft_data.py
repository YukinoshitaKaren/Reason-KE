"""
Generate SFT (Supervised Fine-Tuning) data using GPT-4
This script generates structured reasoning examples for training
"""
import argparse
import json
from tqdm import tqdm
from openai import OpenAI
import os 
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

# Initialize OpenAI client
# You need to set your API key as an environment variable: OPENAI_API_KEY
# For OpenRouter: export OPENAI_API_KEY=your_key_here
# For OpenAI: export OPENAI_API_KEY=your_openai_key

def get_client():
    """Initialize OpenAI client with API key from environment"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # For OpenRouter (alternative to OpenAI)
    base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

SYSTEM = """You are a helpful assistant that can solve the given question step by step with the [Updated Information]. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively.
The reasoning process will be divided into four steps:
<acknowledge>Identify the updated information and the problem to be solved.
<decompose>Break down the original problem into sub problems.
<action>For each sub problem, check if the updated information is relevant and apply the relevant update knowledge, while also considering the previous sub problem. The exact answer is enclosed within \\boxed{} with latex format.
"""

TEMPLE = """Please provide a reasoning process based on my following tasks and corresponding answers.
\n[Task]:\nPlease acknowledge the updated information provided below and respond to the subsequent query.\n\n[Updated Information]:\n[Fact 1]Roblin Park is located in New South Wales.[Fact 2]The Eiffel Tower is located in London.\n\n[Query]:\nWhat is the capital city of the state where Roblin Park is located?
<think>
<acknowledge>The updated information states that Roblin Park is located in New South Wales. And the query is "what is the capital city of the state where Roblin Park is located?"</acknowledge>
<decompose>Break down the original problem into:
[Sub question 1]What state is Roblin Park located in?
[Sub question 2]What is the capital of [sub answer 1]?</decompose>
<action>Answer sub questions based on updated knowledge:
[Sub question 1]Detected relevant to [Fact 1], so the answer is \\boxed{New South Wales}.
[Sub question 2]No relevant facts were detected, but [sub answer 1] can be applied, so the answer is \\boxed{Sydney}.</action>
</think>
<answer>Sydney</answer>

\n[Task]:\n"""

# Template mapping dictionary
TEMPLATE_MAP = {
    'default': TEMPLE,
}

def knowledge_edit_template(new_facts, question):
    """Create knowledge editing prompt template"""
    return "Please acknowledge the updated information provided below and respond to the subsequent query.\n\n[Updated Information]:\n" \
        + new_facts + "\n\n[Query]:\n" + question

def question_template(question):
    """Create simple question template"""
    return "Please respond to the subsequent query.\n\n[Query]:\n" + question

def generate_reasoning(client, query, template='default'):
    """
    Generate reasoning process using GPT-4
    
    Args:
        client: OpenAI client instance
        query: The query to generate reasoning for
        template: Template type to use
        
    Returns:
        Generated reasoning text or None if failed
    """
    max_attempts = 10
    answer = None 
    attempts = 0
    
    while answer is None and attempts < max_attempts:
        try:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": TEMPLATE_MAP.get(template, TEMPLE) + query}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # or "openai/gpt-4o-mini" for OpenRouter
                messages=messages
            )
            
            answer = response.choices[0].message.content
            
        except Exception as e:
            print(f"Exception: {str(e)}")
            print(f"Query: {query}")
            attempts += 1
            if attempts < max_attempts:
                time.sleep(60)  # Wait before retry
            else:
                print(f"Failed after {max_attempts} attempts")
                
    return answer

def process_item(item: Dict[str, Any], client, template: str = 'default') -> Dict[str, Any]:
    """
    Process a single data item, generating reasoning process
    
    Args:
        item: Data item to process
        client: OpenAI client instance
        template: Template type to use
        
    Returns:
        Processed item with generated reasoning or None if failed
    """
    try:
        # Build fact and new question
        fact = f'{item["requested_rewrite"]["prompt"].format(item["requested_rewrite"]["subject"])} {item["requested_rewrite"]["target_new"]["str"]}'
        query = knowledge_edit_template(fact, item['portability']['New Question'])
        
        # Generate reasoning process and answer
        answer = generate_reasoning(client, query, template)
        
        processed_item = item.copy()
        if answer:
            processed_item["answer"] = answer
        
        return processed_item
    except Exception as e:
        print(f"Error processing item: {str(e)}")
        return None

def generate_sft_data_parallel(input_file: str, output_file: str, template: str = 'default', 
                               max_workers: int = 20):
    """
    Generate SFT data in parallel
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        template: Template type to use
        max_workers: Number of parallel workers
    """
    # Initialize client
    client = get_client()
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    total_items = len(data)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(process_item, item, client, template): item 
                         for item in data}
        
        # Show progress with tqdm
        with tqdm(total=total_items, desc="Generating reasoning data") as pbar:
            for future in as_completed(future_to_item):
                try:
                    processed_item = future.result()
                    if processed_item:
                        processed_data.append(processed_item)
                    
                    pbar.update(1)
                    
                    # Save every 10 items
                    if len(processed_data) % 10 == 0:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(processed_data, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f"Error processing data: {str(e)}")
    
    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    print(f"Successfully generated {len(processed_data)} examples")
    print(f"Saved to: {output_file}")

def generate_sft_data_simple(input_file: str, output_file: str, template: str = 'default'):
    """
    Simple non-parallel version for debugging
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        template: Template type to use
    """
    # Initialize client
    client = get_client()
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    
    for item in tqdm(data, desc="Generating reasoning data"):
        fact = f'{item["requested_rewrite"]["prompt"].format(item["requested_rewrite"]["subject"])} {item["requested_rewrite"]["target_new"]["str"]}'
        query = knowledge_edit_template(fact, item['portability']['New Question'])
        
        try:
            # Generate reasoning process and answer
            answer = generate_reasoning(client, query, template)
            
            if answer:
                processed_item = item.copy()
                processed_item["answer"] = answer
                processed_data.append(processed_item)
                
                # Save after each item for debugging
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            continue
    
    print(f"Successfully generated {len(processed_data)} examples")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFT data using GPT-4")
    parser.add_argument("--input", type=str, required=True,
                       help="Input file path (counterfact_portability_gpt4.json)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path")
    parser.add_argument("--template", type=str, default="default",
                       help="Template type to use")
    parser.add_argument("--workers", type=int, default=20,
                       help="Number of parallel workers")
    parser.add_argument("--simple", action="store_true",
                       help="Use simple non-parallel mode for debugging")
    
    args = parser.parse_args()
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it using: export OPENAI_API_KEY=your_key_here")
        exit(1)
    
    # Generate data
    if args.simple:
        generate_sft_data_simple(args.input, args.output, args.template)
    else:
        generate_sft_data_parallel(args.input, args.output, args.template, args.workers)

