# SafeAgent: Towards Safe LLM-based Agents through Agent Constitution

<img width="773" alt="Screen Shot 2024-01-22 at 5 45 55 PM" src="https://github.com/agiresearch/SafeAgent/assets/28013619/5ad4908d-0fa4-4372-9dd1-0281ee7766ae">

## Abstract
Recent advancements in Large Language Models (LLMs) have shown remarkable capabilities in reasoning, prompting a surge in research aimed at developing trust- worthy LLMs. The emergence of LLM-based agents has garnered considerable attention, yet their trustworthiness remains an underexplored area. This aspect is especially critical given the direct interaction of these agents with the physical en- vironment in everyday human activities, placing a premium on their reliability and safety. This paper presents an Agent-Constitution-based framework, SafeAgent, an initial investigation into the improvement of safety dimension of trustworthiness in LLM-based agents. Our findings reveal a concerning deficit in basic safety knowledge and awareness when LLMs function as agents. To address this issue, we propose a framework consisting of threefold strategies: pre-planning enhancement which injects safety knowledge to model prior to plan generation, in-planning enhancement which bolsters safety during plan generation, and post-planning enhancement which ensures safety by post-checking. Through experimental analysis, we demonstrate how these approaches can effectively elevate an LLM agent’s safety by identifying potential challenges. Furthermore, we explore the intricate relationship between an LLM’s general capabilities, such as reasoning, and its efficacy as a safe agent. We argue that a robust reasoning ability is a fundamental prerequisite for an LLM to function safely as an agent. This paper underscores the imperative of integrating safety awareness into the design and deployment of LLM-based agents, not only to enhance their performance but also to ensure their responsible integration into human-centric environments. 

### Agent Constitution: Four key considerations when designing an Agent Constitution

1. Scope of Concern

2. Authorities for Regulation Formation

3. Format of the Constitution

4. Implementation


## Process Diagram for OpenAgent
<img width="773" alt="Screen Shot 2023-10-23 at 12 26 27 PM" src="https://github.com/agiresearch/OpenAgent/assets/28013619/83d267d2-c69e-47c7-af01-df2ac58af646">

<img width="777" alt="Screen Shot 2023-10-21 at 8 54 43 PM" src="https://github.com/agiresearch/OpenAgent/assets/28013619/288d6c85-0b0a-416b-a0fd-620692e96029">

### four agents in the framework: 

**planner**: having access to tools and memory, based on which conduct plan and action

**safety inspector**: having access to safety regulations, based on which regulate the agent's behavior by (1) editing the agent on regulations before planning (2) inform the agent during planning (3) inspect and criticize the agent after planning

**simulator**: having access to the tool definition and agent's action, simulate the observation

**evaluator**: having access to the simulator's outcome, evaluate whether the agent fulfills the user instruction and safety requirement in the end

## QuickStart
### install environment
```
conda create --name safe python=3.9
conda activate safe

git clone https://github.com/dhh1995/PromptCoder
cd PromptCoder
pip install -e .
cd ..

git clone https://github.com/agiresearch/safeAgent.git
cd safeAgent
pip install -r requirements.txt
```

### run experiment
#### set up API keys
If you want to use OpenAI model as base LLM:
```
export OPENAI_API_KEY=your_openai_api_key
```
If you want to use Claude model as base:
```
export CLAUDE_API_KEY=your_claude_api_key
```
#### run command with different parameters
```
domain: medicine, cook, finance, everyday, chemistry
case_idx: {0,1,2,3,4,5,6,7,8,9}
agent_llm_type: gpt-4-1106-preview, gpt-3.5-turbo-1106, claude-2, claude-instant-1.2, mistralai/Mistral-7B-Instruct-v0.1, Qwen/Qwen-72B-Chat 
```
command
```
python main.py --data_dir ../assets/{domain}_data.json --case_idx {case_index} --regulation_check --regulation_prompting --agent_llm_type {agent_llm_type}
```
