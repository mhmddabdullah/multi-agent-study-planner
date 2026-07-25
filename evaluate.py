"""
evaluation/evaluate.py
======================
Evaluation module — runs all 4 variants on test student profiles
and measures quality using DeepEval metrics.

Metrics used (Lab 5):
  - AnswerRelevancyMetric  : does the plan address the student's stated goals?
  - FaithfulnessMetric     : are courses in the plan actually from the catalogue?
  - HallucinationMetric    : did the agent invent courses or prerequisites?

Also records:
  - Generation time per variant
  - Number of revision loops triggered
  - Error count

Outputs:
  - Console table comparing all variants
  - evaluation/results.json for further analysis
  - evaluation/results.csv  for plotting
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq

from graph.pipeline import build_graph, run_pipeline
from variants.configs import ALL_VARIANTS


# ─────────────────────────────────────────────────────────────────────────────
# Wrap ChatGroq as a DeepEval judge model
# ─────────────────────────────────────────────────────────────────────────────

class GroqJudge(DeepEvalBaseLLM):
    """Wraps ChatGroq so DeepEval can use it as a judge model."""

    def __init__(self):
        self.model = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.0,
            max_tokens=1024,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "groq/llama-3.1-8b-instant"


# Test student profiles

TEST_PROFILES = [
    {
        "id": "student_1",
        "description": (
            "I'm a second year CS/AI student. "
            "I'm very interested in deep learning and NLP and want to become an ML engineer. "
            "I struggle with maths and distributed systems. "
            "I prefer project-based assessment. Max 60 credits per semester."
        ),
    },
    {
        "id": "student_2",
        "description": (
            "First year computer science student passionate about cybersecurity and networking. "
            "Strong in programming, weak in statistics. "
            "Want to work in security engineering. "
            "Can handle up to 70 credits a semester."
        ),
    },
    {
        "id": "student_3",
        "description": (
            "Third year AI student. Strong in machine learning and computer vision. "
            "Interested in robotics and autonomous systems. "
            "Weak in databases and web development. "
            "Goal: research scientist. Max 50 credits, prefer exam-based courses."
        ),
    },
]


# Load course catalogue for context strings

_CATALOGUE_PATH = Path(__file__).parent.parent / "data" / "course_catalogue.json"
with open(_CATALOGUE_PATH) as f:
    CATALOGUE = json.load(f)

CATALOGUE_CONTEXT = [
    f"{c['course_code']}: {c['title']}. Topics: {', '.join(c.get('topics', []))}."
    for c in CATALOGUE
]


# Run evaluation

def run_evaluation(variant_keys: list[str] = None, save_results: bool = True) -> dict:
    """
    Run all variants on all test profiles and evaluate with DeepEval.

    Args:
        variant_keys : list of variant keys to run. Defaults to all 4.
        save_results : whether to save results to JSON and CSV files.

    Returns:
        results dict keyed by variant_key -> list of per-student metric dicts.
    """
    if variant_keys is None:
        variant_keys = list(ALL_VARIANTS.keys())

    judge = GroqJudge()
    app = build_graph()
    results = {}

    for variant_key in variant_keys:
        print(f"\n{'='*60}")
        print(f"Running variant: {variant_key} — {ALL_VARIANTS[variant_key].name}")
        print(f"{'='*60}")

        variant_results = []

        for profile in TEST_PROFILES:
            print(f"\n  Student: {profile['id']}")

            # Run pipeline 
            start = time.time()
            try:
                state = run_pipeline(
                    raw_input=profile["description"],
                    variant_key=variant_key,
                    app=app,
                )
                elapsed = time.time() - start
                final_plan = state.get("final_plan", "") or ""
                revision_count = state.get("revision_count", 0)
                errors = state.get("errors", [])
                success = True
            except Exception as e:
                elapsed = time.time() - start
                final_plan = ""
                revision_count = 0
                errors = [str(e)]
                success = False
                print(f"    ERROR: {e}")

            # Build DeepEval test case 
            test_case = LLMTestCase(
                input=profile["description"],
                actual_output=final_plan,
                retrieval_context=CATALOGUE_CONTEXT,
                context=CATALOGUE_CONTEXT,
            )

            # Run metrics
            metric_scores = {}

            if success and final_plan:
                metrics_to_run = [
                    ("answer_relevancy",  AnswerRelevancyMetric(threshold=0.5, model=judge, async_mode=False)),
                    ("faithfulness",      FaithfulnessMetric(threshold=0.5, model=judge, async_mode=False)),
                    ("hallucination",     HallucinationMetric(threshold=0.5, model=judge, async_mode=False)),
                ]

                for metric_name, metric in metrics_to_run:
                    try:
                        metric.measure(test_case)
                        metric_scores[metric_name] = {
                            "score": round(metric.score, 4),
                            "passed": metric.is_successful(),
                            "reason": metric.reason,
                        }
                        print(f"    {metric_name}: {metric.score:.4f} ({'PASS' if metric.is_successful() else 'FAIL'})")
                    except Exception as e:
                        metric_scores[metric_name] = {"score": None, "passed": False, "reason": str(e)}
                        print(f"    {metric_name}: ERROR — {e}")
            else:
                for name in ["answer_relevancy", "faithfulness", "hallucination"]:
                    metric_scores[name] = {"score": 0.0, "passed": False, "reason": "Pipeline failed"}

            variant_results.append({
                "student_id":      profile["id"],
                "variant":         variant_key,
                "success":         success,
                "elapsed_seconds": round(elapsed, 2),
                "revision_count":  revision_count,
                "error_count":     len(errors),
                "errors":          errors,
                "metrics":         metric_scores,
            })

        results[variant_key] = variant_results

    # Print summary table 
    _print_summary(results)

    # Save results 
    if save_results:
        _save_results(results)

    return results


def _print_summary(results: dict):
    print("\n\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Variant':<12} {'Relevancy':>10} {'Faithful':>10} {'Hallucin':>10} {'Time(s)':>9} {'Revisions':>10}")
    print("-"*65)

    for variant_key, variant_results in results.items():
        def avg(metric_name):
            scores = [
                r["metrics"][metric_name]["score"]
                for r in variant_results
                if r["metrics"][metric_name]["score"] is not None
            ]
            return sum(scores) / len(scores) if scores else 0.0

        avg_time = sum(r["elapsed_seconds"] for r in variant_results) / len(variant_results)
        avg_rev  = sum(r["revision_count"]  for r in variant_results) / len(variant_results)

        print(
            f"{variant_key:<12} "
            f"{avg('answer_relevancy'):>10.4f} "
            f"{avg('faithfulness'):>10.4f} "
            f"{avg('hallucination'):>10.4f} "
            f"{avg_time:>9.2f} "
            f"{avg_rev:>10.1f}"
        )


def _save_results(results: dict):
    out_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = out_dir / f"results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # CSV for plotting
    import csv
    csv_path = out_dir / f"results_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant", "student_id", "success",
            "elapsed_seconds", "revision_count", "error_count",
            "answer_relevancy", "faithfulness", "hallucination",
        ])
        for variant_key, variant_results in results.items():
            for r in variant_results:
                writer.writerow([
                    r["variant"],
                    r["student_id"],
                    r["success"],
                    r["elapsed_seconds"],
                    r["revision_count"],
                    r["error_count"],
                    r["metrics"]["answer_relevancy"]["score"],
                    r["metrics"]["faithfulness"]["score"],
                    r["metrics"]["hallucination"]["score"],
                ])
    print(f"CSV saved      → {csv_path}")


if __name__ == "__main__":
    run_evaluation()
