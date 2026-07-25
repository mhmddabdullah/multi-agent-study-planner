"""
agents/schedule_agent.py


Responsibility:
  Takes the student profile and retrieved courses and builds a structured
  semester-by-semester study plan. Respects:
    - Prerequisite ordering (no course before its prerequisites)
    - Credit limits per semester (from student profile)
    - Workload balance (not all heavy courses in one semester)
    - Student weak subjects (spread across semesters, not clustered)

  On revision runs (after Critic feedback), the critique text is included
  in the prompt so the agent can improve its previous draft.

Prompt styles:
  zero_shot      : direct instruction
  few_shot       : example plan shown before the task
  chain_of_thought: step-by-step reasoning before producing JSON

Output written to state:
  draft_plan (dict)
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from graph.state import StudyPlanState
from variants.configs import VariantConfig, get_variant
from utils.llm import get_llm

import time


# Pydantic schema for structured output


class CourseInPlan(BaseModel):
    course_code: str
    title: str
    credits: int
    reason: str = Field(description="Why this course is recommended for this student at this point")


class SemesterPlan(BaseModel):
    year: int
    semester: int
    courses: list[CourseInPlan]
    total_credits: int


class StudyPlan(BaseModel):
    semesters: list[SemesterPlan]
    total_years: int
    notes: str = Field(description="Overall notes about the plan, constraints respected, and any caveats")


# Prompt templates

SYSTEM_BASE = """You are an academic study planner at a university.
Your job is to create a personalised semester-by-semester study plan for a student.

You will receive:
1. The student's profile (interests, year, weak subjects, credit limit, career goal)
2. A list of available courses retrieved from the course catalogue
3. Optionally: critique from a previous plan review (if this is a revision)

Rules you MUST follow:
- Never schedule a course before its prerequisites are completed
- Never exceed the student's max_credits_per_semester
- Spread weak-subject courses across semesters — do not cluster them
- Prioritise courses that match the student's interests and career goal
- Include a reason for each course explaining why it fits this student
- Never schedule courses beyond the student's current year. 
  A Year 3 student's plan covers Year 3 only.
- Include mandatory courses regardless of student interest.

Return ONLY valid JSON with this exact structure — no markdown, no explanation:
{{
  "semesters": [
    {{
      "year": integer,
      "semester": integer,
      "courses": [
        {{
          "course_code": string,
          "title": string,
          "credits": integer,
          "reason": string
        }}
      ],
      "total_credits": integer
    }}
  ],
  "total_years": integer,
  "notes": string
}}"""

SYSTEM_FEW_SHOT_ADDITION = """
Here is an example of a good study plan output (for a different student):

{{
  "semesters": [
    {{
      "year": 1,
      "semester": 1,
      "courses": [
        {{
          "course_code": "CSCI01C",
          "title": "Introduction to Programming",
          "credits": 10,
          "reason": "Foundation course required for all subsequent modules. No prerequisites."
        }},
        {{
          "course_code": "SCIB03C",
          "title": "Mathematics for Computing",
          "credits": 10,
          "reason": "Essential mathematics foundation. Scheduled early to unlock later AI courses."
        }}
      ],
      "total_credits": 20
    }}
  ],
  "total_years": 3,
  "notes": "Maths courses distributed across years 1 and 2 to avoid overload."
}}"""

SYSTEM_COT_ADDITION = """
Before producing the JSON, reason step by step:
1. Which courses are available for the student's current year?
2. Which prerequisites must be scheduled first?
3. Which courses match the student's interests most closely?
4. How should credits be distributed to stay within the semester limit?
5. Are any weak-subject courses clustered? If so, redistribute them.
6. Does the plan align with the student's career goal?

After your reasoning, produce ONLY the JSON output."""


def _build_system_prompt(prompt_style: str) -> str:
    base = SYSTEM_BASE
    if prompt_style == "few_shot":
        base += SYSTEM_FEW_SHOT_ADDITION
    elif prompt_style == "chain_of_thought":
        base += SYSTEM_COT_ADDITION
    return base


USER_TEMPLATE = """Student Profile:
{profile_json}

Available Courses:
{courses_json}

{critique_section}

Build the study plan now."""


# Agent node function


def schedule_agent_node(state: StudyPlanState) -> dict:
    time.sleep(3)  # avoid hitting Groq rate limit
    """
    LangGraph node: Schedule Agent.

    Reads:  state["student_profile"], state["retrieved_courses"],
            state["critique"] (on revision runs), state["variant"]
    Writes: state["draft_plan"]
    """
    config: VariantConfig = get_variant(state["variant"])
    llm = get_llm(config)

    profile = state.get("student_profile", {})
    courses = state.get("retrieved_courses", [])
    critique = state.get("critique", "")

    # Build critique section for the prompt
    if critique:
        critique_section = (
            f"IMPORTANT — Previous plan was reviewed and needs revision.\n"
            f"Critic feedback:\n{critique}\n"
            f"Please address all points above in this revised plan."
        )
    else:
        critique_section = ""

    system_prompt = _build_system_prompt(config.prompt_style)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", USER_TEMPLATE),
    ])

    parser = JsonOutputParser(pydantic_object=StudyPlan)
    chain = prompt | llm | parser

    try:
        plan = chain.invoke({
            "profile_json": json.dumps(profile, indent=2),
            "courses_json": json.dumps(courses, indent=2),
            "critique_section": critique_section,
        })
        return {"draft_plan": plan}

    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"ScheduleAgent error: {str(e)}")
        # Return a minimal fallback plan
        fallback = {
            "semesters": [],
            "total_years": profile.get("year", 3),
            "notes": f"Plan generation failed: {str(e)}",
        }
        return {"draft_plan": fallback, "errors": errors}
