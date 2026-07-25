Multi-Agent Academic Study Planner

A multi-agent system that converts a student's natural-language description of their interests, strengths, weaknesses, and career goals into a personalized, semester-by-semester academic study programme.

Overview:
Rather than a single LLM call, the system routes the task through five specialised agents coordinated with LangGraph, each responsible for one stage of the pipeline. Shared state is passed between nodes, and a conditional reflection loop allows the generated schedule to be critiqued and revised before it's finalised — closer to how a human advisor would draft, review, and adjust a plan rather than producing it in one pass.

Agent Pipeline:
- Profile Extraction — parses the student's free-text description into a structured profile (interests, strengths, weaknesses, career goals)
- Course Retrieval — retrieves relevant courses based on the extracted profile
- Schedule Generation — builds a semester-by-semester plan from the retrieved courses
- Plan Critique — reviews the generated schedule for gaps or misalignment with the student's profile, triggering revision via a reflection loop when needed
- Report Summarisation — produces the final, readable study plan

  
Experimental Variants:
Four variants were tested, differing in prompting strategy and retrieval method:

Variant	Prompting	Retrieval	Reflection
V1	Zero-shot	Keyword-based	None (baseline)
V2	Few-shot	Keyword-based	None
V3	Chain-of-thought	Keyword-based	Two reflection loops (critic ↔ schedule agent)
V4	Zero-shot	Semantic search (Chroma vector store) + cross-encoder re-ranking	None

All variants use Meta Llama 3.1 8B, served via the Groq API.

Evaluation

Evaluated with DeepEval across three metrics: Answer Relevancy, Faithfulness, and Hallucination (threshold 0.5 for all metrics; for Hallucination, lower is better).

Due to Groq free-tier daily token limits, evaluation was conducted across multiple sessions. An initial run tested three student profiles but hit the token limit partway through V2. The scores below are from a completed single representative-profile run across all four variants.

Variant	Answer Relevancy	Faithfulness	Hallucination	Reflection Loops
V1 (Baseline)	1.0000	1.0000	0.0000	0
V2 (Few-Shot)	1.0000	1.0000	0.9111	0
V3 (CoT + Reflect)	1.0000	0.9167	0.8000	2
V4 (RAG)	1.0000	1.0000	0.0000	1


Key Finding:
Answer Relevancy is uniform at 1.0 across all variants — expected, since every variant shares the same profile-extraction agent and course catalogue, so topical alignment is structurally guaranteed regardless of prompting or retrieval strategy.

The Hallucination scores are the most informative result. V1 and V4 both score 0.0 (no hallucinated claims detected), while V2 scores 0.91 and V3 scores 0.80 — worse despite being architecturally more sophisticated. The cause traces to prompting style rather than reasoning depth: V2's few-shot examples teach the summarisation agent to write with more confidence and detail, and it begins asserting claims not grounded in the retrieved courses — most commonly, claiming that weak-subject courses were distributed across semesters when no such courses appeared in the plan at all. V3 shows the same pattern at lower severity, likely because its reflection loops catch some ungrounded claims before finalisation.

V4 uses the same chain-of-thought prompting as V3 but eliminates hallucination entirely. The difference is retrieval: when courses in context are retrieved via semantic search and re-ranked by relevance, the generated plan stays tightly grounded in actual catalogue content, leaving little room for the summarising agent to fabricate. This is the core result of the project — grounding through retrieval quality was a more effective lever for hallucination reduction than prompting sophistication alone.


