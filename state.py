"""
graph/state.py

Defines the shared state that travels through all nodes in the LangGraph.

Every agent reads from this state and returns updates to it.
No agent modifies the state directly — they return a dict of changes,
and LangGraph merges those changes into the current state.
"""

from typing import TypedDict, Optional


class StudyPlanState(TypedDict):
    """
    The single shared state object that flows through the entire pipeline.

    Fields are populated progressively as agents run:
      - raw_input        : set at the start by the user
      - student_profile  : set by Profile Agent
      - retrieved_courses: set by Curriculum Agent
      - draft_plan       : set (and possibly updated) by Schedule Agent
      - critique         : set by Critic Agent
      - revision_count   : incremented each time Schedule Agent revises
      - final_plan       : set by Summarizer
      - variant          : set at the start, read-only throughout
      - errors           : accumulates any non-fatal error messages
    """

    # Input 
    raw_input: str
    """
    The raw text the student typed.
    Example: "I'm a 2nd year CS/AI student interested in deep learning and NLP.
              I struggle with maths. I have 5 free slots per week."
    """

    # Profile Agent output 
    student_profile: Optional[dict]
    """
    Structured student profile extracted by the Profile Agent.
    Keys: major, year, interests, weak_subjects, strong_subjects,
          max_credits_per_semester, career_goal, preferred_assessment_style.
    Example:
    {
        "major": "Computer Science / AI",
        "year": 2,
        "interests": ["deep learning", "NLP", "computer vision"],
        "weak_subjects": ["mathematics", "linear algebra"],
        "strong_subjects": ["programming", "databases"],
        "max_credits_per_semester": 60,
        "career_goal": "ML engineer",
        "preferred_assessment_style": "project-based"
    }
    """

    # Curriculum Agent output 
    retrieved_courses: Optional[list]
    """
    List of course dicts retrieved from the knowledge base.
    Each course has the full structure from course_catalogue.json:
    course_code, title, level, credits, year, semester,
    prerequisites, topics, category, workload_hours, etc.
    """

    # Schedule Agent output 
    draft_plan: Optional[dict]
    """
    The study plan produced by the Schedule Agent.
    Structure:
    {
        "semesters": [
            {
                "year": 1,
                "semester": 1,
                "courses": [
                    {
                        "course_code": "CSCI01C",
                        "title": "Introduction to Programming",
                        "credits": 10,
                        "reason": "Foundational course required before all others."
                    }
                ],
                "total_credits": 50
            },
            ...
        ],
        "total_years": 3,
        "notes": "Prerequisites respected. Maths-heavy courses distributed."
    }
    """

    # Critic Agent output 
    critique: Optional[str]
    """
    The Critic Agent's review of the draft plan.
    Empty string "" means the plan is approved — no revision needed.
    Non-empty means the Schedule Agent should revise.
    Example: "Year 2 Semester 1 is overloaded with 70 credits.
              Deep Learning appears before its prerequisites are met.
              Student's weak maths subjects are all clustered in Year 1."
    """

    revision_count: int
    """
    How many times the Schedule Agent has revised the plan.
    The graph uses this to enforce max_reflection_loops from the variant config.
    Starts at 0, incremented by the Critic node each loop.
    """

    # Summarizer output 
    final_plan: Optional[str]
    """
    The final human-readable study programme produced by the Summarizer.
    A clean, formatted string ready to present to the student.
    """

    # Metadata 
    variant: str
    """
    Which variant is running: "V1", "V2", "V3", or "V4".
    Read by agents to select the correct prompt style and retrieval mode.
    """

    errors: list
    """
    Non-fatal errors accumulated during the run.
    Agents append to this list if something goes wrong but execution continues.
    """
