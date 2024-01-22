import json
import re
from copy import deepcopy
import time
import openai
from safety_regulation.safety_regulation_embedding_computation import (
    get_openai_embedding,
)

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import (
    AgentExecutor,
    BaseMultiActionAgent,
    BaseSingleActionAgent,
    Tool,
    tool,
)
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    BaseRunManager,
    CallbackManager,
    CallbackManagerForChainRun,
)
from langchain.chains import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.input import get_color_mapping
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    BaseStringMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.base_language import BaseLanguageModel
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    AIMessage,
    BaseMessage,
    BaseOutputParser,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool, StructuredTool
from langchain.utilities.asyncio import asyncio_timeout
from procoder.functional import collect_refnames, format_multiple_prompts, format_prompt
from procoder.prompt import Module as PromptModule
from pydantic import BaseModel
from safeagi.prompts.simulator import (
    ADV_SIMULATOR_CRITIQUE,
    ADV_SIMULATOR_CRITIQUE_REPEAT,
    ADV_SIMULATOR_PROMPT,
    ADV_SIMULATOR_SYSTEM_INFO,
    STD_SIMULATOR_CRITIQUE,
    STD_SIMULATOR_CRITIQUE_REPEAT,
    STD_SIMULATOR_PROMPT,
    STD_SIMULATOR_SYSTEM_INFO,
)
from safeagi.prompts.safety_checker import (
    SAFETY_SYSTEM_INFO,
    SAFETY_CHECK_PROMPT,
    SAFETY_PROMPTING_INSTRUCTION
)
from safeagi.prompts.finetune_generator import (
    STD_FINETUNE_GENERATOR_INFO,
    HINDSIGHT_NEGATIVE_EXAMPLE_GENERATOR_INFO,
    SELF_TAUGHT_EXAMPLE_GENERATOR_INFO,
)
from safeagi.tools import RealHuman, RealHumanAssistanceQuery
from safeagi.tools.tool_interface import BaseToolkit
from safeagi.utils import (
    InvalidTool,
    run_with_input_validation,
    validate_outputs,
    Logger,
    hindsight_case_generation,
    create_negative_case_learning_training_file,
    create_positive_case_learning_training_file,
    ordinal,
    finetune_opensource_model
)
from safeagi.utils.my_typing import *

from .agent_executor import AgentExecutorWithToolkit
from .zero_shot_agent_with_toolkit import ZeroShotAgentWithToolkit
import torch
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModel

import os
import sys
import gc 
from numba import cuda
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


from safeagi.utils import (
    get_fixed_model_name,
    load_openai_llm,
    create_training_file,
    run_openai,
    parse_json_string_to_file,
    parse_dict_string
)

Prompt = Union[PromptModule, str]


class SimulatorInputModel(BaseModel):
    simulator_scratchpad: Optional[Any]
    current_tool: Optional[str]
    current_tool_description: Optional[str]
    toolkit_descriptions: Optional[str]
    input: Optional[str]
    underspecifications: Optional[str]
    risky_outcome: Optional[str]
    risky_actions: Optional[str]
    safety_regulations: Optional[str]


# TODO: decouple the simulator from the agent executor
class StandardVirtualAgentExecutorWithToolkit(AgentExecutorWithToolkit):
    # """Virtual agent executor that simulates the execution of virtual tools with LLMs"""
    """Virtual agent executor that outputs thoughts before simulating the execution of virtual tools."""
    logger: Logger
    llm_simulator_chain: LLMChain
    agent_type: str
    base_model_name: str
    toolkits: Sequence[BaseToolkit]
    llm_safety_checker_chain: LLMChain
    llm_critiquer: Optional[BaseLanguageModel] = None
    num_critique_steps: Optional[int] = 0
    max_allowed_steps: Optional[int] = 3
    sim_system_info: Prompt = STD_SIMULATOR_SYSTEM_INFO
    sim_prompt_instruction: Prompt = STD_SIMULATOR_PROMPT
    safety_system_info: Prompt = SAFETY_SYSTEM_INFO
    safety_check_prompt_instruction: Prompt = SAFETY_CHECK_PROMPT
    critique_prompt: Prompt = STD_SIMULATOR_CRITIQUE
    critique_prompt_repeat: Prompt = STD_SIMULATOR_CRITIQUE_REPEAT
    _input_keys: List[str] = ["input"]
    retrieval_tokenizer: Any
    retrieval_model: Any
    regulation_domains: List[str]
    agent_memory: List[str]
    learnt_regulation: List[str]
    domain: str
    case_name: str
    safety_methods: List[str]
    hindsight_memory_directory: str

    @classmethod
    def from_agent_and_toolkits(
        cls,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        agent_type: str,
        toolkits: Sequence[BaseToolkit],
        llm_simulator: BaseLanguageModel,
        llm_safety_checker: BaseLanguageModel,
        retrieval_tokenizer,
        retrieval_model,
        regulation_domains: List[str],
        agent_memory: List[str],
        learnt_regulation: List[str],
        logger,
        domain,
        safety_methods,
        case_name,
        base_model_name,
        hindsight_memory_directory,
        llm_critiquer: Optional[BaseLanguageModel] = None,
        num_critique_steps: Optional[int] = 0,
        max_allowed_steps: Optional[int] = 3,
        callback_manager: Optional[BaseCallbackManager] = None,
        use_chat_format: Optional[bool] = False,
        **kwargs: Any,
    ) -> AgentExecutor:
        """Create from agent and toolkits."""
        print(base_model_name)
        tools = agent.get_all_tools(toolkits)
        tool_names = [tool.name for tool in tools]
        if use_chat_format:
            assert isinstance(llm_simulator, BaseChatModel)

        simulator_prompt = cls.create_simulator_prompt(use_chat_format=use_chat_format)
        llm_simulator_chain = LLMChain(
            llm=llm_simulator,
            prompt=simulator_prompt,
            callback_manager=callback_manager,
        )

        safety_checker_prompt = cls.create_safety_checker_prompt(
            use_chat_format=use_chat_format
        )
        llm_safety_checker_chain = LLMChain(
            llm=llm_safety_checker,
            prompt=safety_checker_prompt,
            callback_manager=callback_manager,
        )

        # NOTE: change to use the simulator as the default critiquer
        if llm_critiquer is None:
            llm_critiquer = llm_simulator

        return cls(
            agent=agent,
            agent_type=agent_type,
            tools=tools,
            toolkits=toolkits,
            tool_names=tool_names,
            llm_simulator_chain=llm_simulator_chain,
            llm_critiquer=llm_critiquer,
            llm_safety_checker_chain=llm_safety_checker_chain,
            num_critique_steps=num_critique_steps,
            max_allowed_steps=max_allowed_steps,
            callback_manager=callback_manager,
            retrieval_tokenizer=retrieval_tokenizer,
            retrieval_model=retrieval_model,
            regulation_domains=regulation_domains,
            agent_memory=agent_memory,
            learnt_regulation=learnt_regulation,
            logger=logger,
            domain=domain,
            case_name=case_name,
            base_model_name=base_model_name,
            safety_methods=safety_methods,
            hindsight_memory_directory=hindsight_memory_directory,
            **kwargs,
        )

    @classmethod
    def get_var(cls, name):
        """Get the default value of a class variable of Pydantic model."""
        return cls.__fields__[name].default

    @classmethod
    def create_simulator_prompt(
        cls, use_chat_format: Optional[bool] = False
    ) -> BasePromptTemplate:
        """Create a the prompt for the simulator LLM."""
        inputs = dict()
        system_info = cls.get_var("sim_system_info")
        prompt_instruction = cls.get_var("sim_prompt_instruction")
        system_info, prompt_instruction = format_multiple_prompts(
            [system_info, prompt_instruction], inputs, include_brackets=[False, True]
        )

        if use_chat_format:
            simulator_system_message = SystemMessage(content=system_info)
            simulator_instruction_message = HumanMessagePromptTemplate.from_template(
                template=prompt_instruction
            )

            messages = [
                simulator_system_message,
                simulator_instruction_message,
            ]
            return ChatPromptTemplate.from_messages(messages=messages)
        else:
            template = "\n\n".join([system_info, prompt_instruction])

            input_variables = cls.get_var("_simulator_input_keys") + [
                "simulator_scratchpad"
            ]
            return PromptTemplate(template=template, input_variables=input_variables)

    @classmethod
    def create_safety_checker_prompt(
        cls, use_chat_format: Optional[bool] = False
    ) -> BasePromptTemplate:
        """Create a the prompt for the simulator LLM."""
        inputs = dict()
        safety_system_info = cls.get_var("safety_system_info")
        safety_system_instruction = cls.get_var("safety_check_prompt_instruction")
        safety_info, safety_check_instruction = format_multiple_prompts(
            [safety_system_info, safety_system_instruction],
            inputs,
            include_brackets=[False, True],
        )

        if use_chat_format:
            safety_system_message = SystemMessage(content=safety_info)
            safety_check_instruction_message = HumanMessagePromptTemplate.from_template(
                template=safety_check_instruction
            )

            messages = [
                safety_system_message,
                safety_check_instruction_message,
            ]
            return ChatPromptTemplate.from_messages(messages=messages)
        else:
            template = "\n\n".join([safety_info, safety_check_instruction])

            input_variables = cls.get_var("_safety_checker_input_keys") + [
                "simulator_scratchpad"
            ]
            return PromptTemplate(template=template, input_variables=input_variables)

    def _get_current_toolkit_descriptions(self, tool_name: str) -> str:
        # NOTE: assume only one toolkit has the tool with tool_name
        for toolkit in self.toolkits:
            for tool in toolkit.tools:
                if tool.name == tool_name:
                    return toolkit.create_description(detail_level="low")
        raise ValueError(f"Tool {tool_name} not found in any of the toolkits.")

    @property
    def simulator_input_keys(self) -> List[str]:
        return self._simulator_input_keys

    @property
    def safety_checker_input_keys(self) -> List[str]:
        return self._safety_checker_input_keys

    @property
    def simulator_generation_prefix(self) -> str:
        return "Simulator Thought: "

    @property
    def safety_checker_generation_prefix(self) -> str:
        return "Safety checker Thought: "

    @property
    def thought_summary_prefix(self) -> str:
        return "Simulator Log Summary: "

    @property
    def stop_seqs(self) -> List[str]:
        return [
            "\nThought:",
            "\n\tThought:",  # or {agent.llm_prefix.rstrip()}
            "\nAction:",
            "\n\tAction:",
        ]

    @property
    def llm_simulator_tool(self) -> BaseTool:
        result = StructuredTool.from_function(
            func=lambda callbacks, **kwargs: self._get_simulated_observation(
                callbacks, **kwargs
            ),
            name="llm_simulator",
            description="Simulate the execution of a tool with a language model",
            args_schema=SimulatorInputModel
            # infer_schema=False
        )
        return result

    def _fix_observation_text(self, text: str):
        return text.rstrip() + "\n"

    def _extract_observation_and_thought(self, llm_output: str) -> Optional[List[str]]:
        """Parse out the observation from the LLM output."""
        # \s matches against tab/newline/whitespace
        regex = rf"{self.thought_summary_prefix}(.*?)[\n]*{self.agent.observation_prefix}[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        if not match:
            return None
        thought_summary = match.group(1).strip()
        observation = match.group(2).strip()
        return observation, thought_summary

    def _get_simulated_observation(
        self, callback_manager: CallbackManager, **full_inputs: Any
    ) -> SimulatedObservation:
        streaming_output = self.llm_simulator_chain.llm.streaming
        if streaming_output:
            print("\n" + self.simulator_generation_prefix)
            # for handler in callback_manager.handlers:
            #     getattr(handler, "on_text")(
            #         "\n" + self.generatetion_prefix, verbose=self.verbose
            #     )
        full_output = self.llm_simulator_chain.predict(
            **full_inputs, stop=self.stop_seqs
        )
        parsed_output = self._extract_observation_and_thought(full_output)
        while parsed_output is None:
            full_output = self._fix_observation_text(full_output)
            full_inputs["simulator_scratchpad"] += full_output
            output = self.llm_simulator_chain.predict(
                **full_inputs, stop=self.stop_seqs
            )
            full_output += output
            parsed_output = self._extract_observation_and_thought(full_output)

        log_output = self.simulator_generation_prefix + full_output
        # remove all the text after self.agent.observation_prefix
        log_output = log_output.split(self.agent.observation_prefix)[0].strip()
        log_output = "\n" + log_output

        if not streaming_output and not log_output.isspace():
            for handler in callback_manager.handlers:
                getattr(handler, "on_tool_end")(log_output, verbose=self.verbose)

        sim_observation = SimulatedObservation(
            observation=parsed_output[0],
            thought_summary=parsed_output[1],
            log=full_output,
        )

        observation = self._critique_simulated_observation(
            callback_manager, sim_observation, full_inputs
        )

        return observation

    def _construct_simulator_scratchpad(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        include_simulator_log: bool = False,
        include_simulator_thought_summary: bool = True,
        include_simulator_last_step_only: bool = False,
    ):
        """Construct the scratchpad that without outputting the last observation."""

        # this is copied from the agent's _construct_scratchpad
        scratchpad = ""
        for idx, (action, observation) in enumerate(intermediate_steps):
            scratchpad += f"Action {idx}: {action.tool}\nAction {idx} Input: {action.tool_input}\n"

            if idx == len(intermediate_steps) - 1:
                scratchpad += "\n"
            else:
                if include_simulator_log and (
                    not include_simulator_last_step_only
                    or idx == len(intermediate_steps) - 2
                ):
                    scratchpad += (
                        f"\n{self.simulator_generation_prefix}{observation.log}\n"
                    )
                elif include_simulator_thought_summary and (
                    not include_simulator_last_step_only
                    or idx == len(intermediate_steps) - 2
                ):
                    scratchpad += f"\n{self.thought_summary_prefix}{observation.thought_summary}\n{self.agent.observation_prefix}{observation.observation}\n"
                else:
                    scratchpad += (
                        f"\n{self.agent.observation_prefix}{observation.observation}\n"
                    )
                # scratchpad += self.agent.llm_prefix

        # add prefix for generation
        scratchpad += self.simulator_generation_prefix
        # scratchpad = self.agent.llm_prefix + scratchpad

        return scratchpad

    def _create_critiquer_prompt(
        self,
        simulator_inputs: Dict[str, str],
        sim_observation: SimulatedObservation,
        critique_outputs: List[Dict[str, str]],
    ) -> BasePromptTemplate:
        """Create a the prompt for the critiquer LLM."""
        refnames = collect_refnames(
            dict(
                sim_prompt=self.sim_prompt_instruction, crit_prompt=self.critique_prompt
            ),
        )
        critique_prompt = format_prompt(
            self.critique_prompt, {}, refnames=refnames, include_brackets=True
        )
        critique_prompt_repeat = format_prompt(
            self.critique_prompt_repeat, {}, refnames=refnames, include_brackets=True
        )

        simulator_prompt_temp = self.llm_simulator_chain.prompt
        use_chat_format = isinstance(simulator_prompt_temp, ChatPromptTemplate)
        simulator_prompt = simulator_prompt_temp.format_prompt(**simulator_inputs)

        critique_prompt_messages = []

        if use_chat_format:
            # add simulator prompt
            critique_prompt_messages += simulator_prompt.messages
        else:
            # add simulator prompt
            critique_prompt_messages.append(HumanMessage(content=simulator_prompt))

        # add simulator output
        simulator_output = sim_observation.log
        critique_prompt_messages.append(AIMessage(content=simulator_output))

        # The last dict in critique_outputs only contains the validation results
        for idx, crit_dict in enumerate(critique_outputs):
            prompt = critique_prompt if idx == 0 else critique_prompt_repeat
            prompt = f"{crit_dict['validation']}\n{prompt}"
            critique_prompt_messages.append(HumanMessage(content=prompt))
            if "critique" in crit_dict:
                # add critique output
                critique_prompt_messages.append(
                    AIMessage(content=crit_dict["critique"])
                )

        if not use_chat_format:
            critique_prompt_messages = "\n\n".join(
                [t.content for t in critique_prompt_messages]
            )

        return critique_prompt_messages

    @property
    def critique_prefix(self) -> str:
        return "Critique #{step}:"

    @property
    def revised_thought_summary_prefix(self) -> str:
        return "Revised Simulator Log Summary #{step}:"

    @property
    def revised_observation_prefix(self) -> str:
        return "Revised Observation #{step}:"

    def _extract_revised_observation_and_thought(
        self, critique_llm_output: str, current_step: int
    ) -> Optional[List[str]]:
        """Parse out the observation from the critiqued LLM output."""
        thought_summary_prefix = self.revised_thought_summary_prefix.format(
            step=current_step
        )
        observation_prefix = self.revised_observation_prefix.format(step=current_step)
        # \s matches against tab/newline/whitespace
        regex = rf"{thought_summary_prefix}(.*?)[\n]*{observation_prefix}[\s]*(.*)"
        match = re.search(regex, critique_llm_output, re.DOTALL)

        if not match:
            return None
        revised_thought_summary = match.group(1).strip()
        revised_observation = match.group(2).strip()
        return revised_observation, revised_thought_summary

    def _critique_simulated_observation(
        self,
        callback_manager: CallbackManager,
        sim_observation: SimulatedObservation,
        simulator_inputs: Dict[str, Any],
    ):
        streaming_output = self.llm_critiquer.streaming

        tool_name = simulator_inputs["current_tool"]
        tool_mapping = dict(zip(self.tool_names, self.tools))
        tool = tool_mapping[tool_name]

        def get_validation_result(obs):
            msg = "The format of the output matches the specification of the tool."
            exception = None
            try:
                outputs = json.loads(obs)
            except json.decoder.JSONDecodeError as e:
                msg = f"The output is not a valid JSON object."
                exception = e
            if exception is None:
                try:
                    validate_outputs(tool.returns, outputs)
                except ValueError as e:
                    msg = f"The format of the output does not match the specification of the tool."
                    exception = e
            return f"Format Validation: {msg}", exception

        current_obs = sim_observation.observation
        critique_outputs = []
        sep = "\n\n"
        revised_output = None

        if self.max_allowed_steps <= 0:
            return sim_observation

        for step in range(self.max_allowed_steps):
            step_idx = step + 1

            validation_msg, exception = get_validation_result(current_obs)
            if exception is not None:
                validation_msg += f" {exception}"
            elif step_idx > self.num_critique_steps:
                # if we have enough number of critique steps and the last output obs is valid
                break

            critique_outputs.append({"validation": validation_msg})
            critiquer_prompt = self._create_critiquer_prompt(
                simulator_inputs, sim_observation, critique_outputs,
            )

            if streaming_output:
                print(f"\n\n{validation_msg}\n\n")
                # for handler in callback_manager.handlers:
                #     getattr(handler, "on_text")("\n\n", verbose=self.verbose)

            crit_out = self.llm_critiquer.generate(
                [critiquer_prompt],
                stop=[
                    self.critique_prefix.format(step=step_idx + 1),
                    "Action:",
                    "Action Input:",
                ],
            )
            assert len(crit_out.generations) == 1
            # todo: this is for chat model
            crit_out = crit_out.generations[0][0].text
            # critique_outputs.append(crit_out)
            critique_outputs[-1]["critique"] = crit_out
            revised_output = self._extract_revised_observation_and_thought(
                crit_out, current_step=step_idx
            )
            current_obs = revised_output[0] if revised_output else current_obs

            log_output = sep + validation_msg + "\n" + crit_out
            if not streaming_output and not log_output.isspace():
                for handler in callback_manager.handlers:
                    getattr(handler, "on_tool_end")(log_output, verbose=self.verbose)

        # todo: extract sim_observation from sim_observation.log
        if revised_output is None:
            return sim_observation

        # todo: the correctness of logging need to be checked.
        logs = [sim_observation.log]
        for crit_dict in critique_outputs:
            logs.append(crit_dict["validation"] + "\n" + crit_dict["critique"])
        log_output_with_critique = sep.join(logs)

        critiqued_observation = SimulatedObservation(
            observation=revised_output[0],
            thought_summary=revised_output[1],
            log=log_output_with_critique,
        )
        # update log in observation
        return critiqued_observation

    def _format_input_for_safety_regulation(self, inputs, intermediate_steps):
        user_instruction = inputs["input"]

        input_string = "User Instruction: {}".format(user_instruction)
        if intermediate_steps:
            input_string += (
                "\nBelow are the actions that have been taken and their result:\n"
            )
            for index, each_action in enumerate(intermediate_steps):
                action = each_action[0]
                action_result = each_action[1]
                tool = action.tool
                tool_input = action.tool_input
                input_string += "\nAction {} is {}".format(index, tool)
                input_string += "\nInputs to Action {} is {}".format(index, tool_input)
                input_string += "\nThe result of Action {} is {}".format(
                    index, action_result
                )

        return input_string

    def _format_extracted_safety_regulation(self, regulations):
        regulations = [
            "({}) ".format(index + 1) + s for index, s in enumerate(regulations)
        ]
        listed_regulations = "\n".join(regulations)
        joined_regulations = (
            "Follow the below safety regulations: \n" + listed_regulations
        )
        safety_prompting_prompt = format_prompt(SAFETY_PROMPTING_INSTRUCTION, {})
        joined_regulations += "\n\n{}".format(safety_prompting_prompt)
        return joined_regulations, regulations

    def _search_for_relevant_memory(self, inputs, intermediate_steps):
        # prototype, empty now
        pass

    def _search_for_relevant_regulation(self, inputs, intermediate_steps):
        input_to_safety_retrieval = self._format_input_for_safety_regulation(
            inputs, intermediate_steps
        )
        if self.retrieval_model is None and self.retrieval_tokenizer is None:
            input_embedding = torch.tensor(
                get_openai_embedding(input_to_safety_retrieval)
            )
        else:
            assert (
                self.retrieval_model is not None
                and self.retrieval_tokenizer is not None
            )

            # Mean pooling for contriever
            def mean_pooling(token_embeddings, mask):
                token_embeddings = token_embeddings.masked_fill(
                    ~mask[..., None].bool(), 0.0
                )
                sentence_embeddings = (
                    token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                )
                return sentence_embeddings
            
            # last token pooling for mistral
            def last_token_pool(last_hidden_states,attention_mask):
                left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                if left_padding:
                    return last_hidden_states[:, -1]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = last_hidden_states.shape[0]
                    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
            def get_detailed_instruct(task_description, query):
                return f'Instruct: {task_description}\nQuery: {query}'

            if 'mistral' not in self.retrieval_model.config._name_or_path:
                # Compute token embeddings
                inputs = self.retrieval_tokenizer(
                    [input_to_safety_retrieval],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                outputs = self.retrieval_model(**inputs)
                input_embedding = mean_pooling(outputs[0], inputs["attention_mask"])
            if 'mistral' in self.retrieval_model.config._name_or_path:
                # Compute token embeddings
                task = 'Given a user instruction, retrieve the most relevant safety regulations'
                input_to_safety_retrieval = [get_detailed_instruct(task, input_to_safety_retrieval)]
                # Apply tokenizer
                max_length = 4096
                # Tokenize the input texts
                batch_dict = self.retrieval_tokenizer(input_to_safety_retrieval, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
                # append eos_token_id to every input_ids
                batch_dict['input_ids'] = [input_ids + [self.retrieval_tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
                inputs = self.retrieval_tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
                outputs = self.retrieval_model(**inputs)
                input_embedding = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

        regulation_embeddings = torch.load(
            "../safety_regulation/safety_regulation_{}_embedding.pt".format(self.domain)
        )

        top = torch.topk(
            torch.cosine_similarity(input_embedding, regulation_embeddings), k=5
        ).indices

        with open("../safety_regulation/safety_regulation.json", "r",) as f:
            regulations = json.load(f)

        regulations = [regulations[self.domain][i] for i in top]

        (
            regulations,
            _format_extracted_safety_regulation,
        ) = self._format_extracted_safety_regulation(regulations)

        return regulations, _format_extracted_safety_regulation

    def reinitialize_agent(self, new_model):
        agent_llm = load_openai_llm(
            model_name=get_fixed_model_name(new_model),
            temperature=0.0,
            request_timeout=300,
            # streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.agent = ZeroShotAgentWithToolkit.from_llm_and_toolkits(
            toolkits=self.toolkits,
            llm=agent_llm,
            agent_type=self.agent_type,
            use_chat_format=isinstance(agent_llm, BaseChatModel),
        )

    def finetune_agent(self, training_file, model_name=None):
        if not model_name:
            model_name = self.base_model_name
        if 'gpt' in model_name:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            training_id = openai.File.create(
                file=open(training_file, "rb"), purpose="fine-tune"
            )["id"]
            openai.FineTuningJob.create(training_file=training_id, model=model_name)
            new_model = None
            while True:
                try:
                    new_model = openai.FineTuningJob.list(limit=1)["data"][0][
                        "fine_tuned_model"
                    ]
                    # either the tuning is done with a new model name generated
                    # or the tuning is failed, and no new model name generated
                    assert new_model is not None or openai.FineTuningJob.list(limit=1)["data"][0]["status"] == "failed"
                    break
                except:
                    time.sleep(0.1)
        else:
            print('--------finetuning opensource model--------')
            destroy_model_parallel()
            del self.agent
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()
            print("Successfully delete the llm pipeline and free the GPU memory!")
            model_name = finetune_opensource_model(training_file, model_name)
        if new_model is None:
            self.logger.log("tuning failed due to insufficient data for tuning.")
            new_model = model_name
        else:
            self.logger.log("tuning done.")
        self.logger.log("tuned model: {}".format(new_model))
        return new_model

    def finetune_regulation(self, relevant_regulations):
        # do not tune on learnt regulations
        new_regulations = [
            r for r in relevant_regulations if r not in self.learnt_regulation
        ]
        training_file_dir = "material/regulation_learning.jsonl"
        if new_regulations:
            self.logger.log("Start preparing regulation learning training data ...")
            self.learnt_regulation += new_regulations
            new_regulations = "\n".join(new_regulations)
            full_output = run_openai(
                STD_FINETUNE_GENERATOR_INFO.format(new_regulations)
            )
            self.logger.log("Creating regulation learning training file ...")
            create_training_file(full_output, training_file_dir)

            self.logger.log("Start regulation tuning ...")
            new_model = self.finetune_agent(training_file_dir)
            self.reinitialize_agent(new_model)
        return new_model

    def finetune_negative_cases(self, criticism_cases, model_name=None):
        training_file_dir = "material/regulation_negative_case_learning_{}_{}.jsonl".format(self.case_name, self.base_model_name)
        self.logger.log(
            "Start preparing regulation negative case learning training data ..."
        )
        while True:
            try:
                full_output = run_openai(
                    hindsight_case_generation(
                        HINDSIGHT_NEGATIVE_EXAMPLE_GENERATOR_INFO, criticism_cases
                    )
                )
                training_file_dir, cases = create_negative_case_learning_training_file(
                    criticism_cases, full_output, training_file_dir
                )  
                self.logger.log("Creating regulation negative case learning training file ...")
                break  
            except:
                time.sleep(0.1)
        self.logger.log("Start regulation negative case learning ...")
        if model_name is None:
            model_name = self.base_model_name
        try:
            new_model = self.finetune_agent(training_file_dir, model_name)
            self.reinitialize_agent(new_model)
        except:
            # when there is not enough data
            new_model = model_name
        return cases, new_model

    def finetune_positive_cases(self, positive_cases, regulations, model_name=None):
        training_file_dir = "material/regulation_positive_case_learning_{}_{}.jsonl".format(self.case_name, self.base_model_name)
        self.logger.log("Start preparing regulation case learning training data ...")
        self.logger.log("Creating regulation case learning training file ...")
        training_file_dir, cases = create_positive_case_learning_training_file(
            positive_cases, regulations, training_file_dir
        )
        self.logger.log("Start regulation case tuning ...")
        if model_name is None:
            model_name = self.base_model_name
        try:
            new_model = self.finetune_agent(training_file_dir, model_name)
            self.reinitialize_agent(new_model)
        except:
            # when there is not enough data
            new_model = model_name
        return cases, new_model

    def append_suggestion(self, inputs, new_suggestion):
        action_trajectories = [s[0].lower() for s in inputs["safety_suggestions"]]
        regulation = [s[1].lower() for s in inputs["safety_suggestions"]]
        if (
            new_suggestion[0].lower() not in action_trajectories
            or new_suggestion[1].lower() not in regulation
        ):
            inputs["safety_suggestions"].append(new_suggestion)
        return inputs
    
    def regulation_postchecking(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        safety_regulations: List[str],
        listed_regulations: str,
        safety_suggestions: List[str],
        model_name: str,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        ):
        # regenerate plan if there's unsafe action
        evaluation_result = {"default": "UNSAFE"}
        mistakes = []
        generation_time = 0
        while set(list((evaluation_result.values()))) != {"SAFE"} and set(list((evaluation_result.values()))) != {"IRRELEVANT"}:
            if generation_time> 0:
                print('=========regenerate plan=========')
                print('=================================')
                print('=================================')
                print('=================================')
            generation_time += 1
            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps, **inputs
            )  # get input +  get next action --> action
            # If the tool chosen is the finishing tool, then we end and return.
            if isinstance(output, AgentFinish):
                return output
            actions: List[AgentAction]
            if isinstance(output, AgentAction):
                actions = [output]
            else:
                actions = output

            for agent_action in actions:
                action_step = len(intermediate_steps) + 1
                if run_manager:
                    run_manager.on_agent_action(
                        agent_action, verbose=self.verbose, color="green"
                    )
                # Otherwise we lookup the tool
                if agent_action.tool in name_to_tool_map:
                    tool = name_to_tool_map[agent_action.tool]
                    return_direct = tool.return_direct
                    color = color_mapping[agent_action.tool]
                    tool_run_kwargs = self.agent.tool_run_logging_kwargs()

                    # We then call the llm to simulate the execution of the tool
                    empty_observation = ""  # for the current step

                    toolkit_descriptions = self._get_current_toolkit_descriptions(
                        agent_action.tool
                    )

                    # for safety checker
                    simulator_scratchpad = self._construct_simulator_scratchpad(
                        intermediate_steps
                        + []
                        + [(agent_action, empty_observation)],
                        include_simulator_thought_summary=True,
                    )
                    current_tool = (
                        "Tool: "
                        + agent_action.tool
                        + "\n Input to the tool: "
                        + agent_action.tool_input
                    )
                    while True:
                        try:
                            (
                                evaluation_result,
                                regulation,
                                criticism,
                            ) = self.single_action_safety_check(
                                safety_regulations=safety_regulations,
                                action_scratchpad=simulator_scratchpad,
                                inputs=inputs["input"],
                                toolkit_description=toolkit_descriptions,
                                current_tool=current_tool,
                            )
                            break
                        except:
                            time.sleep(0.1)

                    # for memory
                    simulator_scratchpad_for_memory = (
                        self._construct_simulator_scratchpad(
                            intermediate_steps
                            + []
                            + [(agent_action, empty_observation)],
                            include_simulator_thought_summary=False,
                        )
                        .replace(self.simulator_generation_prefix, "")
                        .replace("\n\n\nObservation:", "\nObservation:\n")
                        .rstrip("\n")
                    )
                    case_learning_inputs = [
                        inputs,
                        simulator_scratchpad_for_memory,
                        regulation,
                        criticism,
                        action_step,
                    ]
                    if set(list(evaluation_result.values())) != {'SAFE'} and set(list(evaluation_result.values())) != {'IRRELEVANT'}:
                        print("\n")
                        self.logger.log(
                            "!!The current action is deemed as unsafe, RE-PLAN the actions for the {} time!!".format(
                                ordinal(generation_time)
                            )
                        )
                        inputs = self.append_suggestion(
                            inputs,
                            [simulator_scratchpad_for_memory, regulation, criticism],
                        )
                        mistakes.append(simulator_scratchpad_for_memory)
                        if 'regulation_learning' in self.safety_methods:
                            negative_cases, model_name = self.finetune_negative_cases(
                                case_learning_inputs, model_name
                            )
                            self.create_memory(
                                inputs["input"], negative_cases, [regulation], action_step
                            )
                    else:
                        print("\n")
                        self.logger.log("The current action is deemed safe.")
                        if 'regulation_learning' in self.safety_methods:
                            positive_cases, model_name = self.finetune_positive_cases(
                                case_learning_inputs, "\n".join(listed_regulations), model_name
                            )
                            self.create_memory(
                                inputs["input"],
                                positive_cases,
                                listed_regulations,
                                action_step,
                            )
                else:
                    return actions
                
            if  len(mistakes) - len(set(mistakes)) != 0:
                self.logger.log("The agent is consistently making the same errors. \nTo prevent unsafe outcomes and additional costs from API usage, we should temporarily stop the program.")
                #sys.exit("Stopping the program.")
                return AgentFinish({'output': 'no new action generated.'}, 'Final Answer: Plan finished.')

        return actions

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Retrieve relevant safety regulations"""
        self.logger.log("Retrieve relevant safety regulations for the current step ...")
        safety_regulations, listed_regulations = self._search_for_relevant_regulation(
            inputs, intermediate_steps
        )
        if self.safety_methods:
            self.logger.log("Relevant safety regulations for the current step:")
            self.logger.log(safety_regulations)
        inputs["safety_regulations"] = ''
        if 'regulation_prompting' in self.safety_methods:
            inputs["safety_regulations"] = safety_regulations
        model_name = None
        if 'regulation_learning' in self.safety_methods:
            model_name = self.finetune_regulation(listed_regulations)

        """memory searching"""
        # relevant_memories = self._search_for_relevant_memory(inputs, intermediate_steps)
        # inputs['memory'] = relevant_memories
        
        inputs["safety_suggestions"] = []
        if 'regulation_check' in self.safety_methods:
            actions = self.regulation_postchecking(
                name_to_tool_map=name_to_tool_map,
                color_mapping=color_mapping,
                inputs=inputs,
                intermediate_steps=intermediate_steps,
                safety_regulations=safety_regulations,
                listed_regulations=listed_regulations,
                safety_suggestions=inputs["safety_suggestions"],
                model_name=model_name,
                run_manager=run_manager
            )
            if isinstance(actions, AgentFinish):
                return actions
        else:
            output = self.agent.plan(
                intermediate_steps, **inputs
            )  # get input +  get next action --> action
            # If the tool chosen is the finishing tool, then we end and return.
            if isinstance(output, AgentFinish):
                print(output)
                return output
            actions: List[AgentAction]
            if isinstance(output, AgentAction):
                actions = [output]
            else:
                actions = output
            
        result = []

        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(
                    agent_action, verbose=self.verbose, color="green"
                )
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()

                # We then call the llm to simulate the execution of the tool
                empty_observation = ""  # for the current step

                toolkit_descriptions = self._get_current_toolkit_descriptions(
                    agent_action.tool
                )
                # for simulator
                simulator_scratchpad = self._construct_simulator_scratchpad(
                    intermediate_steps + result + [(agent_action, empty_observation)],
                    include_simulator_thought_summary=True,
                )
                simulator_full_inputs = {
                    "simulator_scratchpad": simulator_scratchpad,
                    "current_tool": agent_action.tool,
                    "current_tool_description": tool.description,
                    "toolkit_descriptions": toolkit_descriptions,
                    **inputs,
                }

                observation = run_with_input_validation(
                    self.llm_simulator_tool.run,
                    simulator_full_inputs,
                    tool,
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    **tool_run_kwargs,
                )

                if isinstance(observation, str):
                    observation = SimulatedObservation(
                        observation=observation, thought_summary="", log=observation
                    )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation_text = InvalidTool(available_tools=self.tool_names).run(
                    agent_action.tool,
                    verbose=self.verbose,
                    color=None,
                    **tool_run_kwargs,
                )
                observation = SimulatedObservation(
                    observation=observation_text,
                    thought_summary="",
                    log=observation_text,
                )
            result.append((agent_action, observation))
        return result

    def create_memory(self, user_instruction, cases, regulation, action_step):
        if isinstance(cases, str):
            # positive case
            memory = {
                "domain": self.domain,
                "user_instruction": user_instruction,
                "actions": cases,
                "criticism": "safe",
                "action_step": action_step,
                "regulation": regulation,
            }
            if memory not in self.agent_memory:
                self.agent_memory.append(memory)
        else:
            # negative cases
            for case in cases:
                action = case["action"]
                criticism = case["criticism"]
                memory = {
                    "domain": self.domain,
                    "user_instruction": user_instruction,
                    "actions": action,
                    "criticism": criticism,
                    "action_step": action_step,
                    "regulation": regulation,
                }
                if memory not in self.agent_memory:
                    self.agent_memory.append(memory)

        # all memory for all runnings
        with open(self.hindsight_memory_directory, "a") as f:
            for each_memory in self.agent_memory:
                f.write(json.dumps(each_memory) + "\n")

    def single_action_safety_check(
        self,
        safety_regulations,
        action_scratchpad,
        inputs,
        toolkit_description,
        current_tool,
    ):
        streaming_output = self.llm_safety_checker_chain.llm.streaming
        if streaming_output:
            print("\n" + self.safety_checker_generation_prefix)
            # for handler in callback_manager.handlers:
            #     getattr(handler, "on_text")(
            #         "\n" + self.generatetion_prefix, verbose=self.verbose
            #     )
        inputs = {
            "action_scratchpad": action_scratchpad,
            "input": inputs,
            "safety_regulations": safety_regulations,
            "toolkit_descriptions": toolkit_description,
            "current_tool": current_tool,
        }
        full_output = self.llm_safety_checker_chain.predict(callbacks=None, **inputs)
        evaluation_result, regulation, criticism = self.extract_safety_checking_result(full_output)

        return evaluation_result, regulation, criticism

    def extract_safety_checking_result(self, full_output):
        # first remove irrelevant log
        assert 'Relevance evaluation result' in full_output
        full_output = full_output[full_output.index('Relevance evaluation result'):]
        # start extracting the result
        start_relevance_token = "{"
        end_relevance_token = "}"
        assert start_relevance_token in full_output and end_relevance_token in full_output
        start_relevance_token_index = full_output.index(start_relevance_token)
        end_relevance_token_index = full_output.index(end_relevance_token)
        rest_string = full_output[end_relevance_token_index + 2 :]
        ## extract relevance result
        relevance_evaluation_result = (
            full_output[
                start_relevance_token_index :end_relevance_token_index+1
            ]
            .strip()
            .rstrip("\n")
        )
        relevance_evaluation_result = parse_dict_string(relevance_evaluation_result)
        assert set(list(relevance_evaluation_result.values())).issubset(
                {"RELEVANT", "IRRELEVANT"}
            )
        regulation = ""
        criticism = ""
        evaluation_result = relevance_evaluation_result
        if "RELEVANT" in list(relevance_evaluation_result.values()):
            start_safety_token = "{"
            end_safety_token = "}"
            assert start_safety_token in rest_string and end_safety_token in rest_string
            start_safety_token_index = rest_string.index(start_safety_token)
            end_safety_token_index = rest_string.index(end_safety_token)
            ## extract safety result
            safety_evaluation_result = (
                rest_string[start_safety_token_index : end_safety_token_index+1]
                .strip()
                .rstrip("\n")
            )
            safety_evaluation_result = parse_dict_string(
                safety_evaluation_result
            )
            evaluation_result = safety_evaluation_result
            values = set(list(safety_evaluation_result.values()))
            if values != {"SAFE"}:
                regulation = rest_string[
                    rest_string.index("Regulations that are not satisfied")
                    + len("Regulations that are not satisfied") : rest_string.index(
                        "Criticize the action if necessary"
                    )
                ].strip().replace(':', '')
                criticism = rest_string[
                    rest_string.index("Criticize the action if necessary")
                    + len("Criticize the action if necessary") :
                ].strip().replace(':', '')
        return evaluation_result, regulation, criticism


class AdversarialVirtualAgentExecutorWithToolkit(
    StandardVirtualAgentExecutorWithToolkit
):
    """Adversarial virtual agent executor that simulates the execution of virtual tools that attempt to guide the action agent to make mistakes.
    It will output its thoughts before simualting the execution.
    """

    sim_system_info: Prompt = ADV_SIMULATOR_SYSTEM_INFO
    sim_prompt_instruction: Prompt = ADV_SIMULATOR_PROMPT
    critique_prompt: Prompt = ADV_SIMULATOR_CRITIQUE
    critique_prompt_repeat: Prompt = ADV_SIMULATOR_CRITIQUE_REPEAT
    _input_keys: List[str] = [
        "input",
        "underspecifications",
        "risky_outcome",
        "risky_actions",
    ]