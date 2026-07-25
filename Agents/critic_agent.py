"""
agents/critic_agent.py


Responsibility:
  Reviews the draft study plan produced by the Schedule Agent and identifies
  issues. Implements the Reflection Pattern from Lab 8.

  If issues are found: returns a non-empty critique string.
    → The graph routes back to Schedule Agent for revision.

  If the plan is acceptable (or max revision loops reached): returns "".
    → The graph proceeds to the Summarizer.

  Checks performed:
    1. Prerequisite violations — is any course scheduled before its prerequisites?
    2. Credit overload        — does any semester exceed max_credits_per_semester?
    3. Weak subject clustering — are weak subjects all in the same semester?
    4. Interest alignment     — do the courses actually match the student's interests?
    5. Completeness           — does the plan cover all 3 years?

Output written to state:
  critique (str)       — empty = approved, non-empty = needs revision
  revision_count (int) — incremented by 1 each time this node runs
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph.state import StudyPlanState
from variants.configs import VariantConfig, get_variant
from utils.llm import get_llm

import time


# Prerequisite check (rule-based, no LLM needed)

def _check_prerequisites(plan: dict, all_courses: list[dict]) -> list[str]:
    """
    Check if any course in the plan appears before its prerequisites.
    Returns a list of violation strings.
    """
    # Build map from course_code -> prerequisites
    prereq_map = {c["course_code"]: c.get("prerequisites", []) for c in all_courses}

    # Build ordered list of course codes as they appear in the plan
    scheduled_order = []
    for sem in plan.get("semesters", []):
        for course in sem.get("courses", []):
            scheduled_order.append(course["course_code"])

    violations = []
    seen = set()
    for code in scheduled_order:
        prereqs = prereq_map.get(code, [])
        missing = [p for p in prereqs if p not in seen]
        if missing:
            violations.append(
                f"Course {code} is scheduled before its prerequisites: {missing}"
            )
        seen.add(code)

    return violations


# Credit overload check (rule-based)

def _check_credits(plan: dict, max_credits: int) -> list[str]:
    violations = []
    for sem in plan.get("semesters", []):
        total = sem.get("total_credits", 0)
        if total > max_credits:
            violations.append(
                f"Year {sem['year']} Semester {sem['semester']} has {total} credits, "
                f"exceeding the student's limit of {max_credits}."
            )
    return violations


# LLM-based qualitative critique

SYSTEM_CRITIC = """You are a strict academic plan reviewer.
Your job is to critically evaluate a student's study plan.

Check for:
1. Interest alignment — do the chosen courses actually match the student's stated interests and career goal?
2. Weak subject distribution — are courses the student struggles with spread across semesters, not clustered?
3. Completeness — does the plan make sense for the student's year and goals?
4. Reasoning quality — are the reasons given for each course clear and personalised?

Be specific and constructive. List each issue on a new line.
If the plan looks good and you have no significant issues, respond with exactly: APPROVED

Do not repeat prerequisite or credit issues — those are checked separately.
Do not add any preamble. Start directly with your findings or APPROVED."""

USER_CRITIC = """Student Profile:
{profile_json}

Study Plan to Review:
{plan_json}

Your review:"""

SYSTEM_CRITIC_COT = """You are a strict academic plan reviewer.
Think step by step through each of these checks:
1. Do the courses match the student's interests and career goal?
2. Are weak-subject courses spread out or clustered?
3. Is the plan complete and coherent for this student's year?
4. Is the reasoning for each course clear and personalised?

After your analysis, either:
- List specific issues (be concrete, e.g. "Deep Learning appears in Year 2 Semester 1 but the student is weak in maths which is a prerequisite")
- Or respond with exactly: APPROVED

Do not add preamble. Go directly to your analysis."""


# Agent node function

def critic_agent_node(state: StudyPlanState) -> dict:
    """
    LangGraph node: Critic Agent.

    Reads:  state["draft_plan"], state["student_profile"],
            state["retrieved_courses"], state["revision_count"], state["variant"]
    Writes: state["critique"], state["revision_count"]
    """
    config: VariantConfig = get_variant(state["variant"])
    time.sleep(3)  # avoid hitting Groq rate limit
    llm = get_llm(config)

    plan = state.get("draft_plan", {})
    profile = state.get("student_profile", {})
    all_courses = state.get("retrieved_courses", [])
    revision_count = state.get("revision_count", 0)
    max_loops = config.max_reflection_loops

    #  If max reflection loops reached, approve unconditionally
    if revision_count >= max_loops and max_loops > 0:
        return {"critique": "", "revision_count": revision_count}

    #  Rule-based checks (fast, no LLM) 
    rule_issues = []
    rule_issues += _check_prerequisites(plan, all_courses)
    rule_issues += _check_credits(plan, profile.get("max_credits_per_semester", 60))

    # LLM qualitative review 
    system_prompt = (
        SYSTEM_CRITIC_COT
        if config.prompt_style == "chain_of_thought"
        else SYSTEM_CRITIC
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", USER_CRITIC),
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        llm_review = chain.invoke({
            "profile_json": json.dumps(profile, indent=2),
            "plan_json": json.dumps(plan, indent=2),
        })
    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"CriticAgent LLM error: {str(e)}")
        llm_review = "APPROVED"

    # Combine all issues into final critique 
    all_issues = list(rule_issues)

    if "APPROVED" not in llm_review.upper():
        all_issues.append(llm_review.strip())

    final_critique = "\n".join(all_issues).strip()

    return {
        "critique": final_critique,
        "revision_count": revision_count + 1,
    }
