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
from scorer import score_faithfulness, score_answer_relevancy, score_context_precision

API_URL = os.getenv("API_URL", "http://localhost:8081")
THRESHOLD = 0.7


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

    print("  Scoring faithfulness...")
    faithfulness = score_faithfulness(answer, contexts)
    time.sleep(5)

    print("  Scoring answer relevancy...")
    relevancy = score_answer_relevancy(test_case["question"], answer)
    time.sleep(5)

    print("  Scoring context precision...")
    precision = score_context_precision(test_case["question"], contexts)
    time.sleep(5)

    return {
        "question": test_case["question"],
        "answer": answer,
        "faithfulness": faithfulness["score"],
        "answer_relevancy": relevancy["score"],
        "context_precision": precision["score"],
    }


def main():
    print(f"Running evaluation against {API_URL}...")
    print(f"Total test cases: {len(TEST_QUESTIONS)}\n")

    results = [run_single_evaluation(tc) for tc in TEST_QUESTIONS]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"  Faithfulness:      {r['faithfulness']:.2f}")
        print(f"  Answer relevancy:  {r['answer_relevancy']:.2f}")
        print(f"  Context precision: {r['context_precision']:.2f}")

    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_precision = sum(r["context_precision"] for r in results) / len(results)

    print("\n" + "=" * 70)
    print(f"Average faithfulness:      {avg_faithfulness:.3f}")
    print(f"Average answer relevancy:  {avg_relevancy:.3f}")
    print(f"Average context precision: {avg_precision:.3f}")
    print("=" * 70)

    if avg_faithfulness < THRESHOLD or avg_relevancy < THRESHOLD or avg_precision < THRESHOLD:
        print(f"\n❌ FAILED — one or more metrics below threshold ({THRESHOLD})")
        exit(1)
    else:
        print(f"\n✅ PASSED — all metrics above threshold ({THRESHOLD})")


if __name__ == "__main__":
    main()