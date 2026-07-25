"""
agents/curriculum_agent.py
==========================
Curriculum Agent — retrieves relevant courses from the knowledge base based on the student profile.

Responsibility:
  Retrieves relevant courses from the knowledge base given the student profile.
  Supports 3 retrieval modes controlled by the variant config:

  simple       (V1, V2, V3) — filters by year, category match, and interest keywords.
                               No vector store needed. Fast and deterministic.

  chroma       (V4 base)    — semantic search using HuggingFace embeddings + Chroma.
                               Finds courses whose descriptions are semantically close
                               to the student's interests.

  chroma_rerank (V4)        — chroma + cross-encoder re-ranking for better precision.
                               Retrieves more candidates then re-scores each (query, chunk)
                               pair with a cross-encoder before returning top-k.

Output written to state:
  retrieved_courses (list of course dicts)
"""

import json
import os
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from graph.state import StudyPlanState
from variants.configs import VariantConfig, get_variant

# Load the course catalogue once at module level

_CATALOGUE_PATH = Path(__file__).parent.parent / "data" / "course_catalogue.json"

with open(_CATALOGUE_PATH, "r") as f:
    COURSE_CATALOGUE: list[dict] = json.load(f)


# Mode 1 — Simple retrieval (V1, V2, V3)

def _simple_retrieval(profile: dict, top_k: int) -> list[dict]:
    """
    Filter courses based on:
    - Year eligibility  : course year <= student year + 1 (look one year ahead)
    - Interest matching : any interest keyword appears in course title, topics, or category
    - Weak subject filter: de-prioritise courses whose category matches weak subjects
                           (they appear last in the ranked list)
    Returns top_k courses sorted by relevance score.
    """
    student_year = profile.get("year", 1)
    interests = [i.lower() for i in profile.get("interests", [])]
    weak = [w.lower() for w in profile.get("weak_subjects", [])]

    scored = []
    for course in COURSE_CATALOGUE:
        course_year = course.get("year", 1)

        # Only include courses the student is eligible for or planning ahead
        if course_year > student_year + 1:
            continue

        # Score: count how many interest keywords appear in course text
        text = " ".join([
            course.get("title", ""),
            course.get("category", ""),
            " ".join(course.get("topics", [])),
            course.get("aims", ""),
        ]).lower()

        score = sum(1 for kw in interests if kw in text)

        # Penalise courses that overlap with weak subjects
        weak_penalty = sum(1 for w in weak if w in text)
        score -= weak_penalty * 0.5

        scored.append((score, course))

    # Sort descending by score, return top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# Mode 2 & 3 — Chroma retrieval (V4)

_EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_CHROMA_DIR = Path("/content/chroma_store")
_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    """
    Build or load the Chroma vector store from the course catalogue.
    Called lazily so V1/V2/V3 never load embeddings unnecessarily.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embedding_model = HuggingFaceEmbeddings(
        model_name=_EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Convert each course to a LangChain Document
    docs = []
    for course in COURSE_CATALOGUE:
        text = (
            f"Title: {course['title']}. "
            f"Category: {course['category']}. "
            f"Year: {course['year']}. "
            f"Topics: {', '.join(course.get('topics', []))}. "
            f"Aims: {course.get('aims', '')}."
        )
        docs.append(Document(
            page_content=text,
            metadata={"course_code": course["course_code"], "year": course["year"]},
        ))

    _vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=str(_CHROMA_DIR),
    )
    return _vectorstore


def _chroma_retrieval(profile: dict, top_k: int, rerank: bool = False) -> list[dict]:
    """
    Semantic retrieval using Chroma + HuggingFace embeddings.
    If rerank=True, applies cross-encoder re-ranking after retrieval.
    """
    vs = _get_vectorstore()
    student_year = profile.get("year", 1)
    interests = profile.get("interests", [])
    query = (
        f"Student interested in: {', '.join(interests)}. "
        f"Year {student_year} student looking for relevant courses."
    )

    # Fetch more candidates than needed if re-ranking
    fetch_k = top_k * 2 if rerank else top_k

    results = vs.similarity_search(
        query,
        k=fetch_k,
        filter={"year": {"$lte": student_year + 1}},
    )

    # Map retrieved docs back to full course dicts
    code_to_course = {c["course_code"]: c for c in COURSE_CATALOGUE}
    retrieved = []
    for doc in results:
        code = doc.metadata.get("course_code")
        if code and code in code_to_course:
            retrieved.append(code_to_course[code])

    if not rerank:
        return retrieved[:top_k]

    # Cross-encoder re-ranking
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, doc.page_content) for doc in results]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
    except Exception:
        # If cross-encoder fails, fall back to similarity search order
        return retrieved[:top_k]



# Agent node function

def curriculum_agent_node(state: StudyPlanState) -> dict:
    """
    LangGraph node: Curriculum Agent.

    Reads:  state["student_profile"], state["variant"]
    Writes: state["retrieved_courses"]
    """
    config: VariantConfig = get_variant(state["variant"])
    profile: dict = state.get("student_profile", {})

    try:
        mode = config.retrieval_mode
        top_k = config.top_k_retrieval

        if mode == "simple":
            courses = _simple_retrieval(profile, top_k)

        elif mode == "chroma":
            courses = _chroma_retrieval(profile, top_k, rerank=False)

        elif mode == "chroma_rerank":
            courses = _chroma_retrieval(profile, top_k, rerank=True)

        else:
            raise ValueError(f"Unknown retrieval_mode: {mode}")

        return {"retrieved_courses": courses}

    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"CurriculumAgent error: {str(e)}")
        # Fall back to returning all year-eligible courses
        student_year = profile.get("year", 1)
        fallback = [
            c for c in COURSE_CATALOGUE
            if c.get("year", 1) <= student_year + 1
        ][:config.top_k_retrieval]
        return {"retrieved_courses": fallback, "errors": errors}