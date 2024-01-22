# SafeAgent
SafeAgent: Towards Safe LLM-based Agents through Agent Constitution


<img width="777" alt="Screen Shot 2023-10-21 at 8 54 43 PM" src="https://github.com/agiresearch/OpenAgent/assets/28013619/288d6c85-0b0a-416b-a0fd-620692e96029">



## Process Diagram for OpenAgent
<img width="773" alt="Screen Shot 2023-10-23 at 12 26 27 PM" src="https://github.com/agiresearch/OpenAgent/assets/28013619/83d267d2-c69e-47c7-af01-df2ac58af646">

four agents in the framework: 

**planner**: having access to tools and memory, based on which conduct plan and action

**safety inspector**: having access to safety regulations, based on which regulate the agent's behavior by (1) editing the agent on regulations before planning (2) inform the agent during planning (3) inspect and criticize the agent after planning

**simulator**: having access to the tool definition and agent's action, simulate the observation

**evaluator**: having access to the simulator's outcome, evaluate whether the agent fulfills the user instruction and safety requirement in the end

# QuickStart
## install environment
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

## run experiment
### set up API keys
If you want to use OpenAI model as base LLM:
```
export OPENAI_API_KEY=your_openai_api_key
```
If you want to use Claude model as base:
```
export CLAUDE_API_KEY=your_claude_api_key
```
### run command with different parameters
```
domain: medicine, cook, finance, everyday, chemistry
case_idx: {0,1,2,3,4,5,6,7,8,9}
agent_llm_type: gpt-4-1106-preview, gpt-3.5-turbo-1106, claude-2, claude-instant-1.2, mistralai/Mistral-7B-Instruct-v0.1, Qwen/Qwen-72B-Chat 
```
command
```
python main.py --data_dir ../assets/{domain}_data.json --case_idx {case_index} --regulation_check --regulation_prompting --agent_llm_type {agent_llm_type}
```
