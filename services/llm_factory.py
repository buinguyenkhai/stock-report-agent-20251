"""
LLM Factory for creating model instances via OpenRouter.
"""
from dataclasses import dataclass
from typing import Optional, Type
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import settings
from logger import get_logger

logger = get_logger(__name__)

# OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMConfig:
    """Configuration for LLM calls with task-specific defaults."""
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: int = 120
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_retries: int = 3
    
    @classmethod
    def for_task(cls, task: str) -> "LLMConfig":
        """Get task-specific LLM configuration."""
        task_configs = {
            "item_matching": cls(
                temperature=0.0,
                max_tokens=500,
                timeout=60,
                top_p=0.9,
                frequency_penalty=0.1,  # Reduce repetition in reasoning
                presence_penalty=0.0,
                max_retries=3,
            ),
            "unit_detection": cls(
                temperature=0.0,
                max_tokens=200,
                timeout=60,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_retries=3,
            ),
            "parsing": cls(
                temperature=0.0,
                max_tokens=64000,     # Full financial report parsing needs more tokens
                timeout=600,          # Longer timeout for complex documents
                top_p=0.95,
                frequency_penalty=0.1,
                presence_penalty=0.1,  # Encourage diverse item extraction
                max_retries=3,
            ),
            "query_processing": cls(
                temperature=0.0,
                max_tokens=500,
                timeout=60,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_retries=3,
            ),
            # Single-call extraction of TM tables grouped by referenced notes_ref.
            "notes_tables_by_ref": cls(
                temperature=0.0,
                max_tokens=64000,
                timeout=300,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_retries=3,
            ),
        }
        return task_configs.get(task, cls())


def create_llm(
    model: str = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout: int = 120,
    top_p: float = 0.95,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_retries: int = 3,
    config: Optional[LLMConfig] = None,
    **kwargs
) -> BaseChatModel:
    """
    Factory function to create an LLM instance via OpenRouter.
    """
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Required for OpenRouter models.")
    
    # Use config object if provided, otherwise use individual parameters
    if config is not None:
        temperature = config.temperature
        max_tokens = config.max_tokens
        timeout = config.timeout
        top_p = config.top_p
        frequency_penalty = config.frequency_penalty
        presence_penalty = config.presence_penalty
        max_retries = config.max_retries
    
    model = model or settings.llm_model
    
    logger.debug(f"Creating LLM: model={model}, temp={temperature}, max_tokens={max_tokens}, timeout={timeout}")
    
    # Build default headers for OpenRouter
    default_headers = {
        "HTTP-Referer": "https://github.com/buinguyenkhai/stock-report-agent",
        "X-Title": "Stock Report Agent"
    }
    
    # Merge with any provided headers
    headers = {**default_headers, **kwargs.pop("default_headers", {})}
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        default_headers=headers,
        **kwargs
    )


def create_llm_for_task(
    task: str,
    model: str = None,
    **overrides
) -> BaseChatModel:
    """
    Create an LLM with task-specific configuration.
    """
    config = LLMConfig.for_task(task)
    
    # Apply any overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return create_llm(model=model, config=config)


def create_structured_llm(
    model: str,
    schema: Type[BaseModel],
    temperature: float = 0.0,
    config: Optional[LLMConfig] = None,
    **kwargs
) -> BaseChatModel:
    """
    Create an LLM with structured output (Pydantic model).
    """
    llm = create_llm(model, temperature=temperature, config=config, **kwargs)
    return llm.with_structured_output(schema)


def create_structured_llm_for_task(
    task: str,
    model: str,
    schema: Type[BaseModel],
    **overrides
) -> BaseChatModel:
    """
    Create a structured LLM with task-specific configuration.
    """
    llm = create_llm_for_task(task, model, **overrides)
    return llm.with_structured_output(schema)


def get_model_info(model: str) -> dict:
    """
    Get information about a model for logging/display purposes.
    """
    # Extract just the model name from org/model:variant format
    display_name = model.split("/")[-1].split(":")[0]
    
    return {
        "model": model,
        "provider": "openrouter",
        "display_name": display_name
    }


def test_model_structured_output(
    model: str,
    schema: Type[BaseModel],
    test_prompt: str
) -> tuple[bool, Optional[str]]:
    """
    Test if a model supports structured output with a given schema.
    """
    try:
        llm = create_structured_llm(model, schema, temperature=0.0)
        result = llm.invoke(test_prompt)
        
        # Verify result is valid instance of schema
        if isinstance(result, schema):
            return True, None
        else:
            return False, f"Result is not an instance of {schema.__name__}"
            
    except Exception as e:
        return False, str(e)
