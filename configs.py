"""
variants/configs.py
===================
Defines the 4 experimental variants for the Study Programme Organizer.

V1 — Baseline         : zero-shot prompts, low temperature, no reflection, simple retrieval
V2 — Few-Shot         : few-shot examples injected into prompts, same retrieval as V1
V3 — CoT + Reflection : chain-of-thought prompts, 2 critic reflection loops, mid temperature
V4 — RAG-Enhanced     : full Chroma retrieval + cross-encoder re-ranking, CoT prompts
"""

from dataclasses import dataclass, field
from typing import Literal


PromptStyle  = Literal["zero_shot", "few_shot", "chain_of_thought"]
RetrievalMode = Literal["simple", "chroma", "chroma_rerank"]


@dataclass
class VariantConfig:
    name: str                          # e.g. "V1_Baseline"
    description: str                   # human-readable explanation
    temperature: float                 # LLM temperature
    max_tokens: int                    # max tokens per LLM call
    prompt_style: PromptStyle          # prompting strategy
    retrieval_mode: RetrievalMode      # how Curriculum Agent retrieves courses
    top_k_retrieval: int               # how many courses to retrieve
    max_reflection_loops: int          # how many Critic -> Schedule loops
    model: str = "llama-3.1-8b-instant"  # Groq model to use


# V1 — Baseline
# Zero-shot, low temperature, no reflection, simple keyword retrieval.
# This is the control variant — no special prompting or retrieval tricks.

V1_BASELINE = VariantConfig(
    name="V1_Baseline",
    description=(
        "Zero-shot prompting with low temperature and no reflection loops. "
        "Simple category-based retrieval without a vector store. "
        "Serves as the control/baseline for comparison."
    ),
    temperature=0.1,
    max_tokens=1024,
    prompt_style="zero_shot",
    retrieval_mode="simple",
    top_k_retrieval=14,
    max_reflection_loops=0,
)


# V2 — Few-Shot
# Adds few-shot examples to Profile, Schedule, and Critic prompts.
# Same retrieval and temperature as V1 — isolates the effect of few-shot examples.

V2_FEW_SHOT = VariantConfig(
    name="V2_FewShot",
    description=(
        "Few-shot prompting: example input-output pairs are prepended to "
        "Profile, Schedule, and Critic agent prompts. Same retrieval and "
        "temperature as V1 to isolate the effect of few-shot examples."
    ),
    temperature=0.1,
    max_tokens=1024,
    prompt_style="few_shot",
    retrieval_mode="simple",
    top_k_retrieval=8,
    max_reflection_loops=0,
)


# V3 — Chain-of-Thought + Reflection
# CoT prompts ("think step by step") + 2 Critic reflection loops.
# Higher temperature allows more exploratory reasoning.

V3_COT_REFLECTION = VariantConfig(
    name="V3_CoT_Reflection",
    description=(
        "Chain-of-Thought prompting with 2 Critic reflection loops. "
        "Agents are instructed to reason step by step before producing output. "
        "The Critic reviews and the Schedule Agent revises up to 2 times. "
        "Higher temperature encourages more exploratory course selections."
    ),
    temperature=0.5,
    max_tokens=2048,
    prompt_style="chain_of_thought",
    retrieval_mode="simple",
    top_k_retrieval=10,
    max_reflection_loops=2,
)



# V4 — RAG-Enhanced
# Full Chroma vector store retrieval + cross-encoder re-ranking.
# CoT prompts. Tests whether grounding in semantic search improves plan quality.

V4_RAG_ENHANCED = VariantConfig(
    name="V4_RAG_Enhanced",
    description=(
        "Full RAG pipeline: Chroma vector store with HuggingFace embeddings "
        "for semantic retrieval, followed by cross-encoder re-ranking of candidates. "
        "Chain-of-Thought prompts. Tests whether semantic retrieval over course "
        "descriptions improves relevance compared to simple filtering."
    ),
    temperature=0.3,
    max_tokens=2048,
    prompt_style="chain_of_thought",
    retrieval_mode="chroma_rerank",
    top_k_retrieval=12,
    max_reflection_loops=1,
)


# Registry — import this in other files

ALL_VARIANTS: dict[str, VariantConfig] = {
    "V1": V1_BASELINE,
    "V2": V2_FEW_SHOT,
    "V3": V3_COT_REFLECTION,
    "V4": V4_RAG_ENHANCED,
}


def get_variant(key: str) -> VariantConfig:
    """
    Retrieve a variant config by key (V1, V2, V3, V4).
    Raises ValueError if key is not found.
    """
    if key not in ALL_VARIANTS:
        raise ValueError(
            f"Unknown variant '{key}'. Choose from: {list(ALL_VARIANTS.keys())}"
        )
    return ALL_VARIANTS[key]
