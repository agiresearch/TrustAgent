from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chat_models.base import BaseChatModel
from safeagi.agents import (
    SIMULATORS,
    AgentExecutorWithToolkit,
    ZeroShotAgentWithToolkit,
)
from safeagi.tools import get_toolkits_by_names
from safeagi.utils.my_typing import *
from transformers import AutoTokenizer, AutoModel


def build_agent_executor(
    toolkits: List[str],
    agent_llm: BaseLanguageModel,
    simulator_llm: BaseLanguageModel,
    safety_checker_llm: BaseLanguageModel,
    retrieval_tokenizer,
    retrieval_model,
    regulation_domains,
    agent_memory,
    learnt_regulation,
    logger,
    domain,
    case_name,
    base_model_name,
    safety_methods,
    hindsight_memory_directory,
    critiquer_llm: Optional[BaseLanguageModel] = None,
    num_critique_steps: int = 0,
    max_allowed_steps: int = 3,
    agent_type: str = "naive",
    simulator_type: str = "std_thought",
    verbose: bool = True,
    return_intermediate_steps: bool = True,
    max_iterations: int = 15,
    callback_manager: Optional[BaseCallbackManager] = None,
) -> Type[AgentExecutorWithToolkit]:
    toolkits = get_toolkits_by_names(toolkits)
    agent = ZeroShotAgentWithToolkit.from_llm_and_toolkits(
        toolkits=toolkits,
        llm=agent_llm,
        agent_type=agent_type,
        use_chat_format=isinstance(agent_llm, BaseChatModel),
    )
    executor_class = SIMULATORS[simulator_type]
    if simulator_type != "normal":
        agent_executer = executor_class.from_agent_and_toolkits(
            toolkits=toolkits,
            agent=agent,
            agent_type=agent_type,
            verbose=verbose,
            llm_simulator=simulator_llm,
            llm_safety_checker=safety_checker_llm,
            llm_critiquer=critiquer_llm,
            num_critique_steps=num_critique_steps,
            max_allowed_steps=max_allowed_steps,
            agent_memory=agent_memory,
            learnt_regulation=learnt_regulation,
            use_chat_format=isinstance(simulator_llm, BaseChatModel),
            return_intermediate_steps=return_intermediate_steps,
            max_iterations=max_iterations,
            callback_manager=callback_manager,
            retrieval_tokenizer=retrieval_tokenizer,
            retrieval_model=retrieval_model,
            regulation_domains=regulation_domains,
            domain=domain,
            logger=logger,
            base_model_name=base_model_name,
            safety_methods=safety_methods,
            hindsight_memory_directory=hindsight_memory_directory,
            case_name=case_name,
        )
    else:
        agent_executer = executor_class.from_agent_and_toolkits(
            toolkits=toolkits,
            agent=agent,
            verbose=verbose,
            max_iterations=max_iterations,
            callback_manager=callback_manager,
            retrieval_tokenizer=retrieval_tokenizer,
            retrieval_model=retrieval_model,
            regulation_domains=regulation_domains,
        )
    return agent_executer


# llm_safety_checker
