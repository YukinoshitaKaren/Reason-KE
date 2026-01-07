import re
import sys
import string
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter
from transformers import AutoTokenizer


# --- 新增：引入 sentence-transformers ---
# 为了效率，模型在全局加载一次
try:
    from sentence_transformers import SentenceTransformer, util
    SENT_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except ImportError:
    print("Warning: sentence-transformers not installed. Decomposition quality score will be 0.")
    print("Please run 'pip install sentence-transformers'")
    SENT_MODEL = None

def extract_sub_questions_from_action(action_content: str) -> List[str]:
    """从 action 标签内容中提取所有子问题的文本。"""
    # 这个正则表达式匹配 "[Sub question X]" 之后，到 "\boxed" 之前的所有文本
    pattern = r'\[Sub question \d+\](.*?)(?=\\boxed|\Z)'
    # re.DOTALL 让 '.'可以匹配换行符
    matches = re.findall(pattern, action_content, re.DOTALL)
    # 清理每个匹配项前后的空白字符
    return [match.strip() for match in matches]


def validate_format(text: str) -> Tuple[bool, str]:
    """
    Validate if the text follows the required format with paired tags.
    
    Args:
        text: The text to validate
        
    Returns:
        tuple: (is_valid, reason)
    """
    # Check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"

    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"

    if text.count('<acknowledge>') != text.count('</acknowledge>'):
        return False, "<acknowledge> </acknowledge> not paired"

    if text.count('<acknowledge>') == 0 or text.count('</acknowledge>') == 0:
        return False, "<acknowledge> or </acknowledge> not found"

    if text.count('<decompose>') != text.count('</decompose>'):
        return False, "<decompose> </decompose> not paired"

    if text.count('<decompose>') == 0 or text.count('</decompose>') == 0:
        return False, "<decompose> or </decompose> not found"

    if text.count('<action>') != text.count('</action>'):
        return False, "<action> </action> not paired"

    if text.count('<action>') == 0 or text.count('</action>') == 0:
        return False, "<action> or </action> not found"

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"

    # Check the order of acknowledge/decompose/action
    acknowledge_pos = text.find('<acknowledge>')
    acknowledge_end_pos = text.find('</acknowledge>')
    decompose_pos = text.find('<decompose>')
    decompose_end_pos = text.find('</decompose>')
    action_pos = text.find('<action>')
    action_end_pos = text.find('</action>')
    
    if not (acknowledge_pos < acknowledge_end_pos < decompose_pos < decompose_end_pos < action_pos < action_end_pos):
        return False, "acknowledge/decompose/action tags are not in the correct order"
    

    # Check action content for sub-questions and their answers
    action_start = text.find('<action>')
    action_end = text.find('</action>')
    action_content = text[action_start + 8:action_end]  # +8 to skip '<action>'
    
    # Check if action content contains sub-questions
    if '[Sub question' not in action_content:
        return False, "action section should contain sub-questions"
    
    # Split action content by sub-questions
    sub_question_pattern = r'\[Sub question \d+\]([^[]*?)(?=\[Sub question|\Z)'
    sub_questions = re.findall(sub_question_pattern, action_content, re.DOTALL)
    
    if not sub_questions:
        return False, "no valid sub-questions found in action section"
    
    # Check each sub-question for proper answer format
    for i, sub_content in enumerate(sub_questions, 1):
        sub_content = sub_content.strip()
        if not sub_content:
            return False, f"sub-question {i} has empty content"
        
        # Check if this sub-question has a boxed answer
        if '\\boxed{' not in sub_content or '}' not in sub_content:
            return False, f"sub-question {i} is missing \\boxed{{}} format"
        
        # Ensure the boxed answer is reasonable (not just empty)
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        boxed_matches = re.findall(boxed_pattern, sub_content)
        if not boxed_matches or all(not match.strip() for match in boxed_matches):
            return False, f"sub-question {i} has empty or invalid \\boxed{{}} content"

    return True, "format is correct"


def validate_format_python(text: str) -> Tuple[bool, str]:
    """
    Validate if the text follows the required format for Python with paired tags.
    
    Args:
        text: The text to validate
        
    Returns:
        tuple: (is_valid, reason)
    """
    # Check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"

    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"

    # Check the order of search/result
    current_pos = 0
    while True:
        search_pos = text.find('<python>', current_pos)
        if search_pos == -1:
            break

        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</python>', search_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "python/result tags are incomplete"

        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "python/result tags are nested in the wrong order"

        current_pos = result_end_pos

    # Check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer content from the text within <answer> tags.
    
    Args:
        text: The text to extract answer from
        
    Returns:
        Optional[str]: The extracted answer or None if no match
    """
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    return match.group(1)


def remove_boxed(s: str) -> str:
    """
    Remove the LaTeX \boxed{} wrapper from the string.
    
    Args:
        s: String potentially containing \boxed{}
        
    Returns:
        str: String with \boxed{} removed
    """
    if s is None:
        return ""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extract the last \boxed{} content from the string.
    
    Args:
        string: String to extract \boxed{} from
        
    Returns:
        Optional[str]: The extracted \boxed{} content or None if not found
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def normalize_answer(s: str) -> str:
    """
    Normalize the answer string by removing articles, white spaces, punctuation and converting to lowercase.
    
    Args:
        s: String to normalize
        
    Returns:
        str: Normalized string
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    """
    Calculate F1 score between prediction and ground truths.
    
    Args:
        prediction: The predicted answer
        ground_truths: The ground truth answer(s)
        
    Returns:
        float: F1 score
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])

    return final_metric['f1']

def check_hops(text: str, new_single_hops: List[Dict[str, Any]]) -> Tuple[bool, str]:

    # Check action content for sub-questions and their answers
    action_start = text.find('<action>')
    action_end = text.find('</action>')
    action_content = text[action_start + 8:action_end]  # +8 to skip '<action>'
    
    # Split action content by sub-questions
    sub_question_pattern = r'\[Sub question \d+\]([^[]*?)(?=\[Sub question|\Z)'
    sub_questions = re.findall(sub_question_pattern, action_content, re.DOTALL)


    if len(sub_questions) != len(new_single_hops):
        return False, f"hops wrong: {len(sub_questions)} != {len(new_single_hops)}"

    return True, "hops are correct"

def validate_sub_answer(text: str, new_single_hops: List[Dict[str, Any]]) -> Tuple[int, str]:
    """
    Validate sub-answers and return the count of correct answers.
    
    Returns:
        Tuple[int, str]: (number of correct answers, reason)
    """

    # Check action content for sub-questions and their answers
    action_start = text.find('<action>')
    action_end = text.find('</action>')
    action_content = text[action_start + 8:action_end]  # +8 to skip '<action>'
    
    # Split action content by sub-questions
    sub_question_pattern = r'\[Sub question \d+\]([^[]*?)(?=\[Sub question|\Z)'
    sub_questions = re.findall(sub_question_pattern, action_content, re.DOTALL)

    correct_count = 0
    wrong_details = []

    # Check each sub-question for proper answer format
    for i, (sub_content, new_single_hop) in enumerate(zip(sub_questions, new_single_hops), 1):
        sub_content = sub_content.strip()
        sub_answer = remove_boxed(last_boxed_only_string(sub_content))

        if sub_answer == new_single_hop["answer"] or sub_answer in new_single_hop["answer_alias"]:
            correct_count += 1
        else:
            wrong_details.append(f"sub-question {i} has wrong answer: {sub_answer} != {new_single_hop['answer']} and {sub_answer} not in {new_single_hop['answer_alias']}")

    if correct_count == len(new_single_hops):
        return correct_count, "all sub-questions have correct answer"
    else:
        return correct_count, f"{correct_count}/{len(new_single_hops)} sub-questions correct. " + "; ".join(wrong_details)

def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    一个包含结果分和过程分的结构化奖励函数 (最终版)。
    
    - Outcome Score (max 1.0): 基于最终答案的 F1 分数。
    - Process Score (max 1.0):
        - Hops Score (0.2): Hop 数量是否正确。
        - Decomposition Score (0.4): 模型生成的子问题与标准问题的语义相似度。
        - Sub-answer Score (0.4): 正确的子答案的比例。
    
    Total Score = Outcome Score + Process Score (max 2.0)
    """
    result = {
        "score": 0, "reason": "", "answer": "", "f1_score": 0,
        "outcome_score": 0, "process_score": 0, "hop_score": 0,
        "decomposition_score": 0, "sub_answer_score": 0
    }
    
    # 1. 格式验证
    valid_template, reason = validate_format(solution_str)
    if not valid_template:
        result["score"] = -1.0
        result["reason"] = f"Bad format: {reason}"
        return result
    
    response = solution_str
    
    # 2. 提取最终答案并计算【结果分】
    answer_str = extract_answer(response)
    if answer_str is None:
        result["score"] = -1.0
        result["reason"] = "Cannot extract answer from <answer> tags."
        return result
    
    outcome_score = get_f1_score(answer_str, ground_truth)
    result.update({"answer": answer_str, "f1_score": outcome_score, "outcome_score": outcome_score})

    # 3. 计算【过程分】
    total_hops = len(extra_info.get("new_single_hops", []))
    if total_hops == 0: # 如果问题本身是0跳，过程分为满分1.0
        process_score = 1.0
        result["process_score"] = process_score
        result["score"] = outcome_score + process_score
        result["reason"] = "0-hop question, process score is default to 1.0"
        return result

    # --- a) Hop 数量分 (0.2) ---
    is_hop_count_correct, hop_reason = check_hops(response, extra_info["new_single_hops"])
    hop_score = 0.2 if is_hop_count_correct else 0
    result["hop_score"] = hop_score
    
    # --- b) 子答案分数 (0.4) ---
    correct_sub_count, sub_answer_reason = validate_sub_answer(response, extra_info["new_single_hops"])
    sub_answer_score = (correct_sub_count / total_hops) * 0.4
    result["sub_answer_score"] = sub_answer_score

    # --- c) 分解质量分 (0.4) ---
    decomposition_score = 0
    decomp_reason = "Decomposition quality check skipped."
    if SENT_MODEL is not None and is_hop_count_correct: # 只有hop数对了才检查内容
        action_start = response.find('<action>') + len('<action>')
        action_end = response.find('</action>')
        action_content = response[action_start:action_end]
        
        generated_questions = extract_sub_questions_from_action(action_content)
        ground_truth_questions = [hop['question'] for hop in extra_info['new_single_hops']]
        
        similarities = []
        if len(generated_questions) == len(ground_truth_questions):
            gen_embeddings = SENT_MODEL.encode(generated_questions)
            gt_embeddings = SENT_MODEL.encode(ground_truth_questions)
            
            cos_sim_matrix = util.cos_sim(gen_embeddings, gt_embeddings)
            
            for i in range(len(generated_questions)):
                similarities.append(cos_sim_matrix[i][i].item())
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                decomposition_score = avg_similarity * 0.4
                decomp_reason = f"Avg similarity {avg_similarity:.2f}."
            else:
                decomp_reason = "No similarities to compare."
        else:
             decomp_reason = f"Question count mismatch: generated {len(generated_questions)}, expected {len(ground_truth_questions)}."

    result["decomposition_score"] = decomposition_score

    # 合并过程分
    process_score = hop_score + sub_answer_score + decomposition_score
    result["process_score"] = process_score

    # 4. 计算总分
    final_score = outcome_score + process_score
    result["score"] = final_score
    
    # 5. 生成详细原因
    result["reason"] = (
        f"Final Score: {final_score:.2f} = Outcome({outcome_score:.2f}) + Process({process_score:.2f}). "
        f"Process Breakdown: [Hops: {hop_score:.1f}/0.2, Decomp: {decomposition_score:.2f}/0.4 ({decomp_reason}), SubAns: {sub_answer_score:.2f}/0.4]. "
        f"Sub-ans validation: {sub_answer_reason}."
    )
    
    return result


if __name__ == "__main__":
    # --- 设置 ---
    # 确保 sentence-transformers 模型已加载
    if SENT_MODEL is None:
        print("Cannot run tests without sentence-transformers. Exiting.")
        sys.exit(1)
        
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    answer = "Zoran Milanović"
    extra_info = {
        "tokenizer": tokenizer,
        "new_single_hops": [
            {
                "question": "What country does Ellie Kemper hold citizenship in?",
                "answer": "Croatia",
                "answer_alias": ["CRO", "RH", "HRV", "HR", "Republic of Croatia"]
            },
            {
                "question": "What is the name of the current head of state in Croatia?",
                "answer": "Zoran Milanović",
                "answer_alias": ["Grabar-Kitarović"]
            }
        ]
    }

    # --- 测试案例 ---

    # 案例1：完美的回答 (过程和结果都正确)
    response_perfect = """<think>
<acknowledge>The user wants to know the head of state of the country where Ellie Kemper holds citizenship. The provided context states her citizenship is in Croatia.</acknowledge>
<decompose>
[Sub question 1]What country does Ellie Kemper hold citizenship in?
[Sub question 2]Who is the head of state of [sub answer 1]?</decompose>
<action>
[Sub question 1]What country does Ellie Kemper hold citizenship in? Based on the updated information, the answer is \\boxed{Croatia}.
[Sub question 2]Who is the head of state of Croatia? Based on my knowledge, the answer is \\boxed{Zoran Milanović}.</action>
</think>
<answer>Zoran Milanović</answer>"""

    # 案例2：过程完美，但最终答案错误
    response_process_good_outcome_bad = """<think>
<acknowledge>The user wants to know the head of state of the country where Ellie Kemper holds citizenship. The provided context states her citizenship is in Croatia.</acknowledge>
<decompose>
[Sub question 1]What country does Ellie Kemper hold citizenship in?
[Sub question 2]Who is the head of state of [sub answer 1]?</decompose>
<action>
[Sub question 1]What country does Ellie Kemper hold citizenship in? Based on the updated information, the answer is \\boxed{Croatia}.
[Sub question 2]Who is the head of state of Croatia? Based on my knowledge, the answer is \\boxed{Zoran Milanović}.</action>
</think>
<answer>Someone else</answer>"""

    # 案例3：分解质量差，但子答案和最终答案都“蒙对”了
    response_bad_decomp = """<think>
<acknowledge>The user wants to know something.</acknowledge>
<decompose>
[Sub question 1]What is value 1?
[Sub question 2]What about value 2?</decompose>
<action>
[Sub question 1]Here is the first piece of information. \\boxed{Croatia}.
[Sub question 2]Here is the second. \\boxed{Zoran Milanović}.</action>
</think>
<answer>Zoran Milanović</answer>"""

    # --- 运行和打印结果 ---

    print("--- Case 1: Perfect Response ---")
    res_perfect = compute_score("test", response_perfect, answer, extra_info)
    print(res_perfect['reason'])
    print(f"--> Final Score: {res_perfect['score']:.2f}")

    print("\n--- Case 2: Perfect Process, Wrong Final Answer ---")
    res_process_good = compute_score("test", response_process_good_outcome_bad, answer, extra_info)
    print(res_process_good['reason'])
    print(f"--> Final Score: {res_process_good['score']:.2f}")

    print("\n--- Case 3: Bad Decomposition, Correct Answers (Cheating) ---")
    res_bad_decomp = compute_score("test", response_bad_decomp, answer, extra_info)
    print(res_bad_decomp['reason'])
    print(f"--> Final Score: {res_bad_decomp['score']:.2f}")