from functools import partial
from multiprocessing import Pool

import numpy as np
import tiktoken
import tqdm
from rouge_score import rouge_scorer
import ast 

from .langchain_utils import print_prompt
from .my_typing import *
import json


def parse_json_string_to_file(json_string):
    try:
        # Parse the JSON string
        json_data = json.loads(json_string)
        return json_data

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", str(e))
    except Exception as e:
        print("An error occurred:", str(e))

def parse_dict_string(dict_string):
    try:
        dict_object = ast.literal_eval(dict_string)
        if not isinstance(dict_object, dict):
            raise ValueError("Provided string does not evaluate to a dictionary.")
        return dict_object
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error parsing dictionary string: {e}")

SUFFIXES = {1: "st", 2: "nd", 3: "rd"}


def ordinal(num):
    # I'm checking for 10-20 because those are the digits that
    # don't follow the normal counting scheme.
    if 10 <= num % 100 <= 20:
        suffix = "th"
    else:
        # the second parameter is a default.
        suffix = SUFFIXES.get(num % 10, "th")
    return str(num) + suffix


def get_num_tokens(prompt: str, encoding="cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding)
    return len(encoding.encode(prompt))


def get_toolkit_names(case: Dict[str, Any]) -> List[str]:
    if "Toolkits" in case:
        return case["Toolkits"]
    return case["toolkits"]


def print_intermediate_result_and_stop(results, stop_at):
    if stop_at == "preprocess":
        print(results[0])
    else:
        print_prompt(results[0][0])
    exit(0)


def filter_keys(inputs: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {k: v for k, v in inputs.items() if k in keys}


def check_existence_by_name(new_name: str, existing_names: List[str]) -> bool:
    """Check if the new name is already in the existing names"""
    for name in existing_names:
        if new_name.lower() in name.lower() or name.lower() in new_name.lower():
            return True
    return False


def append_new_items_without_duplicates(
    existing_items: List[Any],
    new_items: List[Any],
    name_func: Callable,
    serialize_func: Callable,
    thresh: float = 0.6,
    num_threads: int = 8,
    pbar: Optional[tqdm.tqdm] = None,
    verbose: bool = False,
) -> List[Any]:
    # TODO: check rouge score is working correctly
    appended_items = []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    tokenize = scorer._tokenizer.tokenize
    all_tokens = [tokenize(serialize_func(x)) for x in existing_items]
    for item in new_items:
        existing_names = [name_func(x) for x in existing_items]
        existing_descs = [serialize_func(x) for x in existing_items]
        name, desc = name_func(item), serialize_func(item)
        if check_existence_by_name(name, existing_names):
            print(f"Discard {name} due to name duplication")
            continue
        new_tokens = tokenize(desc)
        with Pool(num_threads) as p:
            rouge_scores = p.map(
                partial(rouge_scorer._score_lcs, new_tokens), all_tokens
            )
        rouge_scores = [score.fmeasure for score in rouge_scores]

        most_similar_items = {
            existing_descs[i]: rouge_scores[i]
            for i in np.argsort(rouge_scores)[-5:][::-1]
        }
        if verbose:
            print("New item:")
            print(desc)
            print("Most similar items:")
            print(most_similar_items)
            print("Average rouge score:", np.mean(rouge_scores))

        if max(rouge_scores) > thresh:
            print(f"Discard {desc} due to high rouge score {max(rouge_scores)}")
            print(f"most similar items: {most_similar_items}")
            continue

        existing_items.append(item)
        appended_items.append(item)
        all_tokens.append(new_tokens)
        if pbar is not None:
            pbar.update(1)
    return appended_items


def convert_to_score(s, binarize_thres=None):
    v = float(s)
    if binarize_thres is not None:
        v = float(v >= binarize_thres)
    return v
