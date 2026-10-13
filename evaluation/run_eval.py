"""
Custom evaluation runner for the /ask endpoint.

Calls the live /ask endpoint for every question in test_questions.py,
then scores each result using our own Gemini-based judge (scorer.py)
across three dimensions: faithfulness, answer relevancy, and context
precision. Fails the run if any average drops below THRESHOLD —
this is the CI gate.

Usage:
    python run_eval.py

Requires:
    API_URL env var pointing at a running instance of the API
    GEMINI_API_KEY env var (used both by /ask and by the judge)
"""

import os
import time
import httpx

from test_questions import TEST_QUESTIONS
from knowledge_base_mirror import get_texts_for_ids
from scorer import (score_faithfulness, score_answer_relevancy, score_context_precision,
                    score_correctness, score_refusal)

API_URL = os.getenv("API_URL", "http://localhost:8081")
THRESHOLD = 0.7
PAUSE_S = float(os.getenv("EVAL_PAUSE_SECONDS", "5"))   # stay under Gemini rate limits


def call_ask_endpoint(question: str, needs_live_data: bool, **kwargs) -> dict:
    """Call the real /ask endpoint, same as the Streamlit app does."""
    payload = {"question": question}
    if needs_live_data:
        payload["state"] = kwargs.get("state")
        payload["division"] = kwargs.get("division")
        payload["horizon_months"] = kwargs.get("horizon_months", 6)

    response = httpx.post(f"{API_URL}/ask", json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def run_single_evaluation(test_case: dict) -> dict:
    print(f"\nAsking: {test_case['question']}")
    result = call_ask_endpoint(
        test_case["question"],
        test_case["needs_live_data"],
        state=test_case.get("state"),
        division=test_case.get("division"),
        horizon_months=test_case.get("horizon_months", 6),
    )

    answer = result["answer"]
    contexts = get_texts_for_ids(result["retrieved_context"])

    scores = {"question": test_case["question"], "answer": answer,
              "refusal_case": bool(test_case.get("expect_refusal"))}

    print("  Scoring faithfulness...")
    scores["faithfulness"] = score_faithfulness(answer, contexts)["score"]
    time.sleep(PAUSE_S)

    if test_case.get("expect_refusal"):
        print("  Scoring refusal...")
        scores["refusal"] = score_refusal(test_case["question"], answer)["score"]
        time.sleep(PAUSE_S)
        return scores

    print("  Scoring answer relevancy...")
    scores["answer_relevancy"] = score_answer_relevancy(test_case["question"], answer)["score"]
    time.sleep(PAUSE_S)

    print("  Scoring context precision...")
    scores["context_precision"] = score_context_precision(test_case["question"], contexts)["score"]
    time.sleep(PAUSE_S)

    print("  Scoring correctness vs ground truth...")
    scores["correctness"] = score_correctness(test_case["question"], answer, test_case["ground_truth"])["score"]
    time.sleep(PAUSE_S)
    return scores


def main():
    print(f"Running evaluation against {API_URL}...")
    print(f"Total test cases: {len(TEST_QUESTIONS)}\n")

    results = [run_single_evaluation(tc) for tc in TEST_QUESTIONS]

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "correctness", "refusal"]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    for r in results:
        print(f"\nQ: {r['question']}")
        for m in metrics:
            if m in r:
                print(f"  {m:<18} {r[m]:.2f}")

    averages = {}
    for m in metrics:
        vals = [r[m] for r in results if m in r]
        if vals:
            averages[m] = sum(vals) / len(vals)

    print("\n" + "=" * 70)
    for m, v in averages.items():
        print(f"Average {m:<18} {v:.3f}  (n={sum(1 for r in results if m in r)})")
    print("=" * 70)

    failed = [m for m, v in averages.items() if v < THRESHOLD]
    if failed:
        print(f"\n❌ FAILED — below threshold ({THRESHOLD}): {', '.join(failed)}")
        exit(1)
    print(f"\n✅ PASSED — all metrics above threshold ({THRESHOLD})")


if __name__ == "__main__":
    main()
