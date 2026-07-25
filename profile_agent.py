"""
agents/profile_agent.py


Responsibility:
  Takes the raw student input string and extracts a structured profile
  using an LLM with a Pydantic schema for guaranteed structured output.

Prompt styles supported (controlled by variant config):
  - zero_shot      : direct instruction, no examples
  - few_shot       : 2 example input-output pairs before the real input
  - chain_of_thought: asks the model to reason step by step first

Output written to state:
  student_profile (dict)
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from graph.state import StudyPlanState
from variants.configs import VariantConfig, get_variant
from utils.llm import get_llm


# Pydantic schema — defines the structured output we expect

class StudentProfile(BaseModel):
    major: str = Field(description="Student's degree programme, e.g. 'Computer Science / AI'")
    year: int = Field(description="Current year of study (1, 2, or 3)")
    interests: list[str] = Field(description="List of academic or career interests")
    weak_subjects: list[str] = Field(description="Subjects the student finds difficult")
    strong_subjects: list[str] = Field(description="Subjects the student excels at")
    max_credits_per_semester: int = Field(
        description="Maximum credits the student wants per semester (default 60 if not stated)"
    )
    career_goal: str = Field(description="Student's career goal or target job role")
    preferred_assessment_style: str = Field(
        description="Preferred assessment type: project-based, exam-based, or mixed"
    )


# Prompt templates per variant style

SYSTEM_ZERO_SHOT = """You are an academic advisor assistant.
Extract a structured student profile from the student's description.
Return ONLY valid JSON matching this schema exactly — no explanation, no markdown:

{{
  "major": string,
  "year": integer (1-3),
  "interests": [list of strings],
  "weak_subjects": [list of strings],
  "strong_subjects": [list of strings],
  "max_credits_per_semester": integer,
  "career_goal": string,
  "preferred_assessment_style": "project-based" | "exam-based" | "mixed"
}}

If any field is not mentioned, use a sensible default (e.g. year=1, max_credits=60, career_goal="software engineer").
Year mapping: freshman/1st year = 1, sophomore/2nd year = 2, junior/3rd year = 3, senior/final year = 3 (this is a 3-year programme)."""

SYSTEM_FEW_SHOT = """You are an academic advisor assistant.
Extract a structured student profile from the student's description.
Return ONLY valid JSON — no explanation, no markdown.

Here are two examples of correct extractions:

EXAMPLE 1:
Input: "I'm a first year CS student who loves programming and wants to work in web dev. I'm not great at maths."
Output:
{{
  "major": "Computer Science",
  "year": 1,
  "interests": ["programming", "web development"],
  "weak_subjects": ["mathematics"],
  "strong_subjects": ["programming"],
  "max_credits_per_semester": 60,
  "career_goal": "web developer",
  "preferred_assessment_style": "project-based"
}}

EXAMPLE 2:
Input: "Third year AI student. Strong in deep learning and NLP, weak in distributed systems. Want to do research. Prefer exams. Max 50 credits a semester."
Output:
{{
  "major": "Artificial Intelligence",
  "year": 3,
  "interests": ["deep learning", "NLP", "research"],
  "weak_subjects": ["distributed systems"],
  "strong_subjects": ["deep learning", "NLP"],
  "max_credits_per_semester": 50,
  "career_goal": "AI researcher",
  "preferred_assessment_style": "exam-based"
}}

Year mapping: freshman/1st year = 1, sophomore/2nd year = 2, junior/3rd year = 3, senior/final year = 3 (this is a 3-year programme).
Now extract the profile from the student's input below. Return ONLY valid JSON."""


SYSTEM_COT = """You are an academic advisor assistant.
Extract a structured student profile from the student's description.

Think step by step:
1. Identify the student's degree programme and year
2. List all interests mentioned (explicit and implied)
3. Identify weak subjects (struggles, dislikes, avoids)
4. Identify strong subjects (enjoys, excels, mentions positively)
5. Infer career goal from interests and context
6. Determine credit load preference (default 60 if not stated)
7. Determine assessment preference (default "mixed" if not stated)

Year mapping: freshman/1st year = 1, sophomore/2nd year = 2, junior/3rd year = 3, senior/final year = 3 (this is a 3-year programme).

After reasoning, return ONLY valid JSON — no markdown, no extra text:
{{
  "major": string,
  "year": integer,
  "interests": [list of strings],
  "weak_subjects": [list of strings],
  "strong_subjects": [list of strings],
  "max_credits_per_semester": integer,
  "career_goal": string,
  "preferred_assessment_style": "project-based" | "exam-based" | "mixed"
}}"""


PROMPT_STYLE_MAP = {
    "zero_shot": SYSTEM_ZERO_SHOT,
    "few_shot": SYSTEM_FEW_SHOT,
    "chain_of_thought": SYSTEM_COT,
}


# Agent node function

def profile_agent_node(state: StudyPlanState) -> dict:
    """
    LangGraph node: Profile Agent.

    Reads:  state["raw_input"], state["variant"]
    Writes: state["student_profile"]

    Returns a dict with the updated fields (LangGraph merges this into state).
    """
    config: VariantConfig = get_variant(state["variant"])
    llm = get_llm(config)

    system_prompt = PROMPT_STYLE_MAP[config.prompt_style]

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Student input:\n{raw_input}"),
    ])

    parser = JsonOutputParser(pydantic_object=StudentProfile)

    chain = prompt | llm | parser

    try:
        profile = chain.invoke({"raw_input": state["raw_input"]})
        return {"student_profile": profile}

    except Exception as e:
        # Non-fatal: return a default profile and log the error
        default_profile = {
            "major": "Computer Science",
            "year": 1,
            "interests": [],
            "weak_subjects": [],
            "strong_subjects": [],
            "max_credits_per_semester": 60,
            "career_goal": "software engineer",
            "preferred_assessment_style": "mixed",
        }
        errors = state.get("errors", [])
        errors.append(f"ProfileAgent error: {str(e)}")
        return {"student_profile": default_profile, "errors": errors}
