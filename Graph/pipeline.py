"""
graph/pipeline.py

Wires all 5 agent nodes into a LangGraph StateGraph.


Graph flow:
  START
    → profile_agent
    → curriculum_agent
    → schedule_agent
    → critic_agent
        ↓
    [needs_revision?]
        ├── "revise"   → schedule_agent   (loop back)
        └── "approve"  → summarizer
    → END

The conditional edge after critic_agent checks:
  - Is the critique non-empty? (issues found)
  - AND has revision_count NOT exceeded max_reflection_loops?
  → If both true: route back to schedule_agent
  → Otherwise: route to summarizer
"""

from langgraph.graph import StateGraph, START, END

from graph.state import StudyPlanState
from agents.profile_agent import profile_agent_node
from agents.curriculum_agent import curriculum_agent_node
from agents.schedule_agent import schedule_agent_node
from agents.critic_agent import critic_agent_node
from agents.summarizer import summarizer_node
from variants.configs import get_variant


# Conditional edge: should we revise or approve?

def should_revise(state: StudyPlanState) -> str:
    """
    Routing function called after the Critic node.

    Returns "revise"  → routes back to schedule_agent
    Returns "approve" → routes forward to summarizer
    """
    config = get_variant(state["variant"])
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    # Approve if: no issues found, OR max loops reached
    if not critique or revision_count >= config.max_reflection_loops:
        return "approve"

    return "revise"


# Build and compile the graph

def build_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph and return the compiled app.
    Call this once and reuse the returned app for all variant runs.
    """
    builder = StateGraph(StudyPlanState)

    # Register nodes
    builder.add_node("profile_agent",     profile_agent_node)
    builder.add_node("curriculum_agent",  curriculum_agent_node)
    builder.add_node("schedule_agent",    schedule_agent_node)
    builder.add_node("critic_agent",      critic_agent_node)
    builder.add_node("summarizer",        summarizer_node)

    # Fixed edges 
    builder.add_edge(START,              "profile_agent")
    builder.add_edge("profile_agent",    "curriculum_agent")
    builder.add_edge("curriculum_agent", "schedule_agent")
    builder.add_edge("schedule_agent",   "critic_agent")

    # Conditional edge after critic 
    builder.add_conditional_edges(
        "critic_agent",
        should_revise,
        {
            "revise":  "schedule_agent",   # loop back for revision
            "approve": "summarizer",        # proceed to final output
        },
    )

    #  Final edge 
    builder.add_edge("summarizer", END)

    return builder.compile()



# Convenience runner

def run_pipeline(
    raw_input: str,
    variant_key: str,
    app=None,
) -> StudyPlanState:
    """
    Run the full pipeline for a given student input and variant.

    Args:
        raw_input   : The student's natural language description.
        variant_key : "V1", "V2", "V3", or "V4".
        app         : Pre-compiled graph (optional). If None, builds a new one.

    Returns:
        The final StudyPlanState after all nodes have run.
    """
    if app is None:
        app = build_graph()

    initial_state: StudyPlanState = {
        "raw_input":         raw_input,
        "student_profile":   None,
        "retrieved_courses": None,
        "draft_plan":        None,
        "critique":          "",
        "revision_count":    0,
        "final_plan":        None,
        "variant":           variant_key,
        "errors":            [],
    }

    final_state = app.invoke(initial_state)
    return final_state
