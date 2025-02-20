import torch
import re
import random
from symeval import EvaluatorMathBatch
# from latex2sympy2_extended import NormalizationConfig
# from math_verify import LatexExtractionConfig, StringExtractionConfig, ExprExtractionConfig, parse, verify




def extract_answer_math(s):
    ans = s.split("boxed")
    if len(ans) == 1:
        return s
    ans = ans[-1]
    if len(ans) == 0:
        return ""
    try:
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
    except:
        return ""
    return a


def calculate_accuracy_reward(predictions, solutions, do_print=False):
    """Reward function that checks if the completion is the same as the ground truth."""
    pred_answers = [extract_answer_math(pred) for pred in predictions]

    evaluator = EvaluatorMathBatch()
    scores = evaluator.batch_eq(ref_answers=solutions, pred_answers=pred_answers)
    
    for i, pred in enumerate(predictions):
        if "boxed" not in pred:
            scores[i] = -0
        else:
            if not scores[i]:
                scores[i] = -1.0
            else:
                scores[i] = 1.0

    if do_print:
        for pred, sol, score, pred_all in zip(pred_answers, solutions, scores, predictions):
            print(f"Pred: {pred}  <===>  Sol: {sol}  <===>  Score: {score}")
            print(f"Pred All: {pred_all}")
    return scores

# case = '<think>123</think><think>123</think><answer>456</answer>'
def is_format_correct(completion):
    # pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    pattern = r"^<think>.*?</think>"
    if not re.match(pattern, completion, re.DOTALL | re.MULTILINE):
        return False
    # check if all tags only appear once
    # tags = ["<think>", "</think>", "<answer>", "</answer>"]
    tags = ["<think>", "</think>"]
    for tag in tags:
        if completion.count(tag) != 1:
            return False
    
    # check if <think>...</think> is empty
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, completion, re.DOTALL | re.MULTILINE)
    if think_match and think_match.group(1).strip() == "":
        return False
    
    return True

def calculate_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    completion_contents = completions
    return [1.0 if is_format_correct(content) else -1.0 for content in completion_contents]


def extract_qwen_output(prompt):
    return prompt.split("<｜Assistant｜>")[-1].strip()


def reward_func(queries, prompts, **kwargs):
    # queries is prompts + responses
    answers = kwargs['answers']
    responses = [extract_qwen_output(query) for query in queries]

    do_print = False
    if random.randint(0, 5) == 1:
        do_print = True
        
    if do_print:
        print(f"Response Case: {responses[0]}")

    format_rewards = calculate_format_reward(responses)
    accuracy_rewards = calculate_accuracy_reward(responses, answers, do_print=do_print)

    final_rewards = []
    for format_reward, accuracy_reward in zip(format_rewards, accuracy_rewards):
        final_rewards.append(accuracy_reward + 0.05 * format_reward)
        # if accuracy_reward == 0.0:
        #     # will skip this example in reinforce algo
        #     final_rewards.append(0.0)
        # elif accuracy_reward == 1.0 and format_reward == 1.0:
        #     final_rewards.append(1.0)
        # elif accuracy_reward == -1.0 and format_reward == 1.0:
        #     final_rewards.append(-0.5)
        # elif accuracy_reward == 1.0 and format_reward == -1.0:
        #     final_rewards.append(0.5)
        # else:
        #     final_rewards.append(-1.0)
    
    return torch.tensor(final_rewards)