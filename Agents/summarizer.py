"""
agents/summarizer.py

Responsibility:
  Takes the approved draft_plan dict and the student_profile and produces
  a clean, human-readable study programme document.

  The output is a formatted string — not JSON — that could be shown
  directly to a student or printed as a report.

  Always uses low temperature (hardcoded 0.1) regardless of variant,
  because we want consistent, deterministic formatting at this final stage.

Output written to state:
  final_plan (str)
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

from graph.state import StudyPlanState
from variants.configs import VariantConfig, get_variant


# Prompt

SYSTEM_SUMMARIZER = """You are an academic advisor writing a personalised study programme report for a student.

Given the student's profile and their approved study plan (in JSON), write a clean,
well-structured, human-readable report.

The report must include:
1. A personalised introduction addressing the student by their major and career goal
2. A semester-by-semester breakdown with:
   - All courses listed clearly (code, title, credits)
   - A brief explanation of why each course was chosen for this student
3. A summary section highlighting:
   - How the plan addresses the student's interests
   - How weak subjects are handled across the programme
   - Any important notes or caveats
4. A closing encouragement paragraph

Write in a professional but friendly tone. Use clear headings.
Do not output JSON — output a formatted text report."""

USER_SUMMARIZER = """Student Profile:
{profile_json}

Approved Study Plan:
{plan_json}

Write the study programme report now:"""


# Agent node function

def summarizer_node(state: StudyPlanState) -> dict:
    """
    LangGraph node: Summarizer.

    Reads:  state["draft_plan"], state["student_profile"], state["variant"]
    Writes: state["final_plan"]
    """
    config: VariantConfig = get_variant(state["variant"])

    # Summarizer always uses low temperature for consistent formatting
    llm = ChatGroq(
        model=config.model,
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=2048,
    )

    profile = state.get("student_profile", {})
    plan = state.get("draft_plan", {})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_SUMMARIZER),
        ("user", USER_SUMMARIZER),
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        final_plan = chain.invoke({
            "profile_json": json.dumps(profile, indent=2),
            "plan_json": json.dumps(plan, indent=2),
        })
        return {"final_plan": final_plan}

    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"Summarizer error: {str(e)}")
        # Fallback: convert plan dict to plain text
        fallback = f"Study Plan Summary\n\n{json.dumps(plan, indent=2)}"
        return {"final_plan": fallback, "errors": errors}
