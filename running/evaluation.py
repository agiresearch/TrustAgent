import json
import random
import re
from argparse import ArgumentParser, Namespace
import sys 
import os
sys.path.append('../')

from langchain.schema import ChatResult, LLMResult
from procoder.functional import removed_submodules
from procoder.prompt import Module as PromptModule
from procoder.functional import format_multiple_prompts, format_prompt
from procoder.functional import indent4 as indent
from procoder.functional import removed_submodules, replaced_submodule
from procoder.prompt import *
from safeagi.utils import get_num_tokens, make_colorful


from safeagi.prompts.evaluator import (
    AGENT_HELP_EVALUATOR_PROMPT,
    AGENT_HELP_EVALUATOR_SYSTEM_INFO,
    AGENT_SAFE_EVALUATOR_PROMPT,
    AGENT_SAFE_EVALUATOR_SYSTEM_INFO,
    AGENT_PROGRESS_EVALUATOR_SYSTEM_INFO, 
    AGENT_PROGRESS_EVALUATOR_PROMPT,
)
from safeagi.prompts.globals import GENERAL_CRITIQUE_PROMPT
from safeagi.tools import get_toolkits_by_names
from safeagi.utils import (
    METRICS_SHORT2LONG,
    case_to_input_dict,
    construct_trajec,
    get_toolkit_names,
    get_used_tools_in_trajec,
    parse_llm_response,
)
from safeagi.utils.my_typing import *
from safeagi.utils.my_typing import Any, Dict, List

from safeagi.executors import BasePromptExecutorWithCritique


import time
import anthropic
import openai

def run_openai(input_sentence, max_tokens_to_sample: int = 5000):
    output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "user", "content": input_sentence}],
        max_tokens=max_tokens_to_sample,
        temperature=0.2,
    )

    output = output.choices[0].message["content"].strip()

    return output

def load_data_in_evaluation(save_file_path, case_dir):
    results = []
    with open(save_file_path, 'r') as f:
        for l in f:
            results.append(json.loads(l))
    result = results[0]

    with open(case_dir, 'r') as f:
        all_inputs = json.load(f)

    case_idx = result['case_idx']
    input_case = [input_case for input_case in all_inputs if input_case['name'] == case_idx][0]
    toolkits = get_toolkits_by_names(get_toolkit_names(input_case))
    toolkit_descs = {
        detail_level: "\n".join(
            [f"{toolkit.create_description(detail_level)}" for toolkit in toolkits]
        )
        for detail_level in ["low", "medium", "high"]
    }
    prompt_inputs = dict(
        tool_names=", ".join([tool.name for tool in toolkits]),
        toolkit_descriptions=toolkit_descs['high'],
        ground_truth=input_case["Ground Truth"],
    )
    prompt_postinputs = {
        "input":input_case["User Instruction"],
        "underspecifications":'\n'+'\n'.join([k + ': ' + ' '.join(v) for k,v in input_case['Underspecifications'].items()]),
        "expected_achievements":'\n'+' '.join(input_case['Expected Achievements']),
        "evaluator_scratchpad":result['trajectory'],
        "ground_truth": input_case["Ground Truth"],
        "risky_outcome": input_case["Potential Risky Outcomes"],
        "risky_actions": input_case["Potential Risky Actions"],
        "ground_truth": input_case["Ground Truth"],
        }
    
    return prompt_inputs, prompt_postinputs


def compute_helpfulness_result(prompt_inputs, prompt_postinputs):
    system_prompt, example_prompt = format_multiple_prompts(
        [AGENT_HELP_EVALUATOR_SYSTEM_INFO, AGENT_HELP_EVALUATOR_PROMPT,],
        prompt_inputs,
        include_brackets=[False, True],
    )
    system_prompt = system_prompt
    human_prompt = example_prompt.format(**prompt_postinputs)

    def parse(output):
        label_token = 'Overall Qualitative Label:'
        score_token = 'Overall Quantitative Score:'
        label_string = output[output.find(label_token) + len(label_token):]
        label = label_string[:label_string.index('\n')].strip().rstrip('\n')
        score_string = output[output.find(score_token) + len(score_token):]
        score = score_string[:score_string.index('\n')].strip().rstrip('\n')
        return label, score

    output = run_openai(system_prompt + human_prompt)
    while True:
        try:
            label, score = parse(output)
            break 
        except:
            output = run_openai(system_prompt + human_prompt)

    return label, score

def compute_safe_result(prompt_inputs, prompt_postinputs):
    system_prompt, example_prompt = format_multiple_prompts(
        [AGENT_SAFE_EVALUATOR_SYSTEM_INFO, AGENT_SAFE_EVALUATOR_PROMPT,],
        prompt_inputs,
        include_brackets=[False, True],
    )
    system_prompt = system_prompt
    human_prompt = example_prompt.format(**prompt_postinputs)

    def parse(output):
        label_token = 'Overall Qualitative Label:'
        score_token = 'Overall Quantitative Score:'
        label_string = output[output.find(label_token) + len(label_token):]
        label = label_string[:label_string.index('\n')].strip().rstrip('\n')
        score_string = output[output.find(score_token) + len(label_token):]
        score = score_string[:score_string.index('\n')].strip().rstrip('\n')
        return label, score

    output = run_openai(system_prompt + human_prompt)
    while True:
        try:
            label, score = parse(output)
            break 
        except:
            output = run_openai(system_prompt + human_prompt)

    return label, score 


def compute_progress_result(prompt_inputs, prompt_postinputs):
    system_prompt, example_prompt = format_multiple_prompts(
        [AGENT_PROGRESS_EVALUATOR_SYSTEM_INFO, AGENT_PROGRESS_EVALUATOR_PROMPT,],
        prompt_inputs,
        include_brackets=[False, True],
    )
    system_prompt = system_prompt
    human_prompt = example_prompt.format(**prompt_postinputs)

    def parse(output):
        output = output + '\n'
        print('-------progress output-------')
        print(output)
        counting_token = 'Total Correct Calls Counting:'
        counting_proportion_token = 'Total Correct Calls Proportion:'
        prefix_token = 'Length of Correct Prefix in Plan:'
        prefix_proportion_token = 'Proportion of Correct Prefix in Plan:'

        counting_token_index = [i for i, c in enumerate(output) if output[i:i+len(counting_token)] == counting_token][-1]
        counting_string = output[counting_token_index + len(counting_token):]
        counting = counting_string[:counting_string.index('\n')].strip().rstrip('\n')

        counting_proportion_index = [i for i, c in enumerate(output) if output[i:i+len(counting_proportion_token)] == counting_proportion_token][-1]
        counting_proportion_string = output[counting_proportion_index + len(counting_proportion_token):]
        counting_proportion = counting_proportion_string[:counting_proportion_string.index('\n')].strip().rstrip('\n')

        prefix_token_index = [i for i, c in enumerate(output) if output[i:i+len(prefix_token)] == prefix_token][-1]
        prefix_string = output[prefix_token_index + len(prefix_token):]
        prefix = prefix_string[:prefix_string.index('\n')].strip().rstrip('\n')

        prefix_proportion_index = [i for i, c in enumerate(output) if output[i:i+len(prefix_proportion_token)] == prefix_proportion_token][-1]
        prefix_proportion_string = output[prefix_proportion_index + len(prefix_proportion_token):]
        prefix_proportion = prefix_proportion_string[:prefix_proportion_string.index('\n')].strip().rstrip('\n')
        
        return counting, counting_proportion, prefix, prefix_proportion

    output = run_openai(system_prompt + human_prompt)
    number = 10
    times = 0
    while True:
        try:
            times += 1
            counting, counting_proportion, prefix, prefix_proportion = parse(output)
            _ = int(counting)
            _ = int(prefix)
            break 
        except:
            output = run_openai(system_prompt + human_prompt)
        if times > number:
            break

    return counting, counting_proportion, prefix, prefix_proportion

def construct_result(prompt_inputs, prompt_postinputs):
    helpfulness_label, helpfulness_score = compute_helpfulness_result(prompt_inputs, prompt_postinputs)
    safe_label, safe_score = compute_safe_result(prompt_inputs, prompt_postinputs)
    counting, counting_proportion, prefix, prefix_proportion = compute_progress_result(prompt_inputs, prompt_postinputs)

    result = {
        'helpfulness_label': helpfulness_label,
        'helpfulness_score': helpfulness_score,
        'safe_label': safe_label,
        'safe_score': safe_score,
        'counting': counting,
        'counting_proportion': counting_proportion,
        'prefix': prefix,
        'prefix_proportion': prefix_proportion,
    }

    return result

if __name__ == '__main__':
    save_file_path = 'results/trajectory/gpt-4-1106-preview_medicine_0_.jsonl'
    case_dir = '../assets/medicine_data.json'
    prompt_inputs, prompt_postinputs = load_data_in_evaluation(save_file_path, case_dir)

    result = construct_result(prompt_inputs, prompt_postinputs)
    print(result)

    # counting, counting_proportion, prefix, prefix_proportion = compute_progress_result(prompt_inputs, prompt_postinputs)
    # print((counting, counting_proportion, prefix, prefix_proportion))
