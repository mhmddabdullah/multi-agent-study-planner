"""
utils/llm.py
============
Single LLM factory function used by all agents.

All agents call get_llm(config) to get a ChatGroq instance
configured for the current variant. This ensures:
  - Consistent model settings across the pipeline
  - Easy swapping of model or provider in one place
  - Temperature and token limits controlled by variant config
"""

import os
from langchain_groq import ChatGroq
from variants.configs import VariantConfig


def get_llm(config: VariantConfig) -> ChatGroq:
    """
    Return a ChatGroq LLM instance configured for the given variant.

    The GROQ_API_KEY must be set as an environment variable before calling this.
    In Colab: os.environ["GROQ_API_KEY"] = "gsk_..."
    In .env file: GROQ_API_KEY=gsk_...

    Args:
        config: The VariantConfig for the current run.

    Returns:
        A ChatGroq instance ready to use in LCEL chains or LangGraph nodes.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set.\n"
            "In Colab, run: os.environ['GROQ_API_KEY'] = 'gsk_your_key_here'"
        )

    return ChatGroq(
        model=config.model,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
