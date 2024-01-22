import sys
import openai

sys.path.append("..")
import os
import json
from argparse import Namespace
from functools import partial
import tiktoken
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import load_dotenv

load_dotenv()
from safeagi.agent_executor_builder import build_agent_executor
from safeagi.utils import (
    construct_trajec,
    construct_simple_trajec,
    append_file,
    get_fixed_model_name,
    load_openai_llm,
    get_toolkit_names,
    case_to_input_dict,
    read_file,
    make_colorful,
    print_prompt,
)
from safeagi.utils import append_jsonl, replace_agent_action_with_list, Logger
from transformers import AutoTokenizer, AutoModel
from safety_regulation.safety_regulation_embedding_computation import (
    embedding_computation_contriever,
    embedding_computation_openai,
)
from evaluation import construct_result, load_data_in_evaluation
import datetime
import argparse

now = datetime.datetime.now()
time_string = now.strftime("%Y-%m-%d_%H:%M:%S")
logging_dir = "log/test_{}.log".format(time_string)
logger = Logger(logging_dir, True)


def create_parser():
    parser = argparse.ArgumentParser()
    # setting
    parser.add_argument('--agent_llm_type', type=str, default="gpt-3.5-turbo-1106", help='base model for the agent, choose from ["gpt-4", "gpt-3.5-turbo-16k", "claude-2", "claude-1", "mistralai/Mistral-7B-Instruct-v0.1", "internlm/internlm-chat-20b"]')
    parser.add_argument('--agent_type', type=str, default='naive', help='type of agent with different prompts, choose from ["naive", "ss_only", "helpful_ss"]')
    parser.add_argument('--agent_temp', type=float, default=0.0, help='agent temperature')
    parser.add_argument('--simulator_llm_type', type=str, default='gpt-4', help='base model for the emulator, we fix it to gpt-4 for the best emulation performance')
    parser.add_argument('--safety_checker_llm_type', type=str, default='gpt-4', help='base model for the safety checker, currently using GPT series for simple implementation')
    parser.add_argument('--simulator_type', type=str, default='adv_thought', help='emulator type, choose from ["std_thought", "adv_thought"] for standrd or adversarial emulation')
    parser.add_argument('--use_retriever', type=str, default='contriever', help='retriever type, choose from ["contriever", "mistral", "openai"] for different retriever')
    parser.add_argument('--regulation_prompting', action='store_true', help='whether to use regulation prompting')
    parser.add_argument('--regulation_check', action='store_true', help='whether to use regulation check')
    parser.add_argument('--regulation_learning', action='store_true', help='whether to use regulation learning')

    # data
    parser.add_argument('--case_idx', type=int, default=0, help='case id for the test case')
    parser.add_argument('--data_dir', type=str, default='../assets/medicine_data.json', help='path to the test case')

    # evaluation only
    parser.add_argument('--eval_only', action='store_true', help='only do evaluation')
    parser.add_argument('--save_file_path', type=str, default='results/trajectory/medicine_1_regulation_prompting_2024-01-13_19:13:42.jsonl', help='path to the saved trajectory')

    args = parser.parse_args()

    return args
        

if __name__ == "__main__":
    args = create_parser()

    show_prompt = False

    agent_llm_type = args.agent_llm_type # "gpt-4"  # base model for the agent, choose from ["gpt-4", "gpt-3.5-turbo-16k", "claude-2", "claude-1", "mistralai/Mistral-7B-Instruct-v0.1", "internlm/internlm-chat-20b"]
    agent_type = args.agent_type  # type of agent with different prompts, choose from ["naive", "ss_only", "helpful_ss"]
    agent_temp = args.agent_temp  # agent temperature
    simulator_llm_type =  args.simulator_llm_type  # base model for the emulator, we fix it to gpt-4 for the best emulation performance
    safety_checker_llm_type = args.safety_checker_llm_type # "gpt-3.5-turbo-16k"  # base model for the safety checker, currently using GPT series for simple implementation
    simulator_type = args.simulator_type  # emulator type, choose from ["std_thought", "adv_thought"] for standrd or adversarial emulation
    logger.log("-------print settings-------")
    logger.log("::::::print agent_llm: {}::::::".format(agent_llm_type))
    logger.log("::::::print simulator_llm: {}::::::".format(simulator_llm_type))
    logger.log("::::::print safety_checker_llm: {}::::::".format(safety_checker_llm_type))
    logger.log("::::::print simulator_type: {}::::::".format(simulator_type))


    ## Select from our curated test cases
    case_dir = args.data_dir
    cases = read_file(case_dir)
    case_idx = args.case_idx # Choose your case id here
    case = cases[case_idx]
    domain = case['name'].split('_')[0]

    if args.eval_only and args.save_file_path:
        domain = args.save_file_path.split('/')[-1].split('_')[1]
        case_idx = int(args.save_file_path.split('/')[-1].split('_')[2])
        case_dir = '../assets/{}_data.json'.format(domain)
        cases = read_file(case_dir)
        case = cases[case_idx]
    else:
        case_dir = args.data_dir
        cases = read_file(case_dir)
        case_idx = args.case_idx # Choose your case id here
        case = cases[case_idx]
        domain = case['name'].split('_')[0]

    ## safety methods
    safety_methods = []
    if  args.regulation_prompting:
        safety_methods.append("regulation_prompting")
    if args.regulation_check:
        safety_methods.append("regulation_check")
    if args.regulation_learning:
        safety_methods += ["regulation_learning", "regulation_check"]
    safety_methods = list(set(safety_methods))
    if 'gpt-4' in agent_llm_type or 'claude' in agent_llm_type:
        # these models do not support regulation learning
        if 'regulation_learning' in safety_methods:
            safety_methods.remove('regulation_learning')
    logger.log("::::::print safety_methods: {}::::::".format(str(safety_methods)))

    if not args.eval_only:
        # ######### planning agent #########
        # The planning LLM
        agent_llm = load_openai_llm(
            model_name=get_fixed_model_name(agent_llm_type),
            temperature=agent_temp,
            request_timeout=300,
            # streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        # The emulator LLM
        simulator_llm = load_openai_llm(
            model_name=get_fixed_model_name(simulator_llm_type),
            temperature=0.0,
            request_timeout=300,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        # The safety checker LLM
        safety_checker_llm = load_openai_llm(
            model_name=get_fixed_model_name(safety_checker_llm_type),
            temperature=0.0,
            request_timeout=300,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        encoding = tiktoken.get_encoding("cl100k_base")

        # safety regulation retriever
        if args.use_retriever == 'contriever':
            retrieval_tokenizer = AutoTokenizer.from_pretrained(
                "facebook/contriever-msmarco", cache_dir="pretrained_models",
            )
            retrieval_model = AutoModel.from_pretrained(
                "facebook/contriever-msmarco", cache_dir="pretrained_models",
            )
            #regulation_domains = ['chemistry', 'cooking', 'finance', 'medicine', 'general_domain']
            regulation_domains = embedding_computation_contriever(
                retrieval_tokenizer,
                retrieval_model,
                "../safety_regulation/safety_regulation.json",
                "../safety_regulation/",
            )
        elif args.use_retriever == 'mistral':
            retrieval_tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct', cache_dir='pretrained_models/')
            retrieval_model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', cache_dir='pretrained_models/')
            #regulation_domains = ['chemistry', 'cooking', 'finance', 'medicine', 'general_domain']
            regulation_domains = embedding_computation_contriever(
                retrieval_tokenizer,
                retrieval_model,
                "../safety_regulation/safety_regulation.json",
                "../safety_regulation/",
            )
        else:
            regulation_domains = embedding_computation_openai(
                "../safety_regulation/safety_regulation.json", "../safety_regulation/",
            )
            retrieval_tokenizer = None
            retrieval_model = None

        ######### setting basic components
        agent_memory = []
        learnt_regulation = []
        hindsight_memory_directory = "material/hindsight_memory.jsonl"
        build_agent_executor = partial(
            build_agent_executor,
            agent_llm=agent_llm,
            agent_type=agent_type,
            simulator_llm=simulator_llm,
            safety_checker_llm=safety_checker_llm,
            retrieval_tokenizer=retrieval_tokenizer,
            retrieval_model=retrieval_model,
            regulation_domains=regulation_domains,
            agent_memory=agent_memory,
            learnt_regulation=learnt_regulation,
            logger=logger,
            domain=domain,
            base_model_name=agent_llm_type,
            safety_methods = safety_methods,
            hindsight_memory_directory=hindsight_memory_directory,
            case_name = case['name'],
        )

        def query_agent(case, simulator_type="std_thought", max_iterations=15):
            agent_executer = build_agent_executor(
                get_toolkit_names(case),
                simulator_type=simulator_type,
                max_iterations=max_iterations,
            )
            del case['Ground Truth']
            prompt_inputs = case_to_input_dict(case)
            if "adv" in simulator_type:
                return agent_executer(prompt_inputs)
            else:
                return agent_executer(prompt_inputs["input"])

        def display_prompt(prompt):
            print(make_colorful("human", prompt.split("Human:")[1]))

        ######### planning agent prompt #########
        toolkit = get_toolkit_names(case)
        agent_executor = build_agent_executor(
            toolkits=toolkit, simulator_type=simulator_type,
        )

        agent_prompt_temp = agent_executor.agent.llm_chain.prompt
        agent_prompt = agent_prompt_temp.format(
            **{k: "test" for k in agent_prompt_temp.input_variables}
        )
        if show_prompt:
            display_prompt(agent_prompt)
            print("\n\n>>>>Token lengths:", len(encoding.encode(agent_prompt)))

        simulator_prompt_temp = agent_executor.llm_simulator_chain.prompt
        simulator_prompt = simulator_prompt_temp.format(
            **{k: "test" for k in simulator_prompt_temp.input_variables}
        )
        if show_prompt:
            display_prompt(simulator_prompt)
            print("\n\n>>>>Token lengths:", len(encoding.encode(simulator_prompt)))


        ######### actual agent planning + simulation start here #########
        print("-----------actually running experiments-----------")
        results = query_agent(case=case, simulator_type=simulator_type)
        simplified_traj = construct_simple_trajec(results)

        def save_traj(path, simplified_traj):
            results= {}
            # This is an ad-hoc fix for dumping langchain result
            sim_type = "Standard" if simulator_type == "std_thought" else "Adversarial"
            results["sim_type"] = sim_type
            results["agent_llm"] = agent_llm_type
            results["agent_temp"] = agent_temp
            results["case_idx"] = case['name']
            results["trajectory"] = simplified_traj
            with open(path, 'a') as f:
                results_string = json.dumps(results)
                f.write(results_string + '\n')

        agent_llm_type = agent_llm_type.replace('/', '_')
        args.save_file_path = "results/trajectory/{}_{}_{}.jsonl".format(agent_llm_type,case['name'], '_'.join(safety_methods))
        if os.path.exists(args.save_file_path):
            args.save_file_path = "results/trajectory/{}_{}_{}_{}.jsonl".format(agent_llm_type,case['name'], '_'.join(safety_methods), time_string)
        save_traj(args.save_file_path, simplified_traj)

    ######### perform evaluation here #########
    print("-----------start running evaluation-----------")
    prompt_inputs, prompt_postinputs = load_data_in_evaluation(args.save_file_path, case_dir)
    openai.api_key = os.environ["OPENAI_API_KEY"]
    result = construct_result(prompt_inputs, prompt_postinputs)
    print(result)
    
    save_result_path = args.save_file_path.replace('trajectory', 'score')
    with open(save_result_path, 'w') as f:
        json.dump(result, f)

