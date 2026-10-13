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
                    score_correctness, score_refusal, JudgeUnavailable, RETRY_WAITS_S)

API_URL = os.getenv("API_URL", "http://localhost:8081")
THRESHOLD = 0.7
PAUSE_S = float(os.getenv("EVAL_PAUSE_SECONDS", "5"))   # stay under Gemini rate limits
# Stop early when Gemini is clearly down, instead of retrying every call:
MAX_CONSECUTIVE_UNAVAILABLE = int(os.getenv("EVAL_MAX_CONSECUTIVE_UNAVAILABLE", "3"))
MAX_MINUTES = float(os.getenv("EVAL_MAX_MINUTES", "30"))     # whole-run budget (a slow-but-working Gemini day took ~22 min)


class Abort(RuntimeError):
    """Gemini/API unavailable too often, or the time budget ran out."""


_state = {"consecutive_unavailable": 0, "started": time.time()}


def _check_budget():
    if (time.time() - _state["started"]) / 60 > MAX_MINUTES:
        raise Abort(f"time budget of {MAX_MINUTES:.0f} min exceeded")


def call_ask_endpoint(question: str, needs_live_data: bool, **kwargs) -> dict:
    """Call the real /ask endpoint, same as the Streamlit app does."""
    payload = {"question": question}
    if needs_live_data:
        payload["state"] = kwargs.get("state")
        payload["division"] = kwargs.get("division")
        payload["horizon_months"] = kwargs.get("horizon_months", 6)

    # /ask itself calls Gemini, so it can fail transiently too (5xx), and it is
    # rate-limited (429). Retry those; anything else is a real failure.
    last = None
    for wait in RETRY_WAITS_S + [None]:
        try:
            response = httpx.post(f"{API_URL}/ask", json=payload, timeout=60.0)
            if response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()
                return response.json()
            last = f"HTTP {response.status_code}"
        except httpx.TimeoutException as e:
            last = f"timeout ({e.__class__.__name__})"
        if wait is None:
            break
        print(f"  /ask unavailable ({last}), retrying in {wait}s")
        time.sleep(wait)
    raise JudgeUnavailable(f"/ask kept failing: {last}")


def _try(score_fn, *args):
    """A metric's score, or None if Gemini stayed unavailable after retries."""
    _check_budget()
    try:
        score = score_fn(*args)["score"]
        _state["consecutive_unavailable"] = 0
        return score
    except JudgeUnavailable as e:
        _state["consecutive_unavailable"] += 1
        print(f"  !! skipped {score_fn.__name__}: Gemini unavailable ({e})")
        if _state["consecutive_unavailable"] >= MAX_CONSECUTIVE_UNAVAILABLE:
            raise Abort(f"Gemini unavailable for {MAX_CONSECUTIVE_UNAVAILABLE} calls in a row")
        return None


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
    scores["faithfulness"] = _try(score_faithfulness, answer, contexts)
    time.sleep(PAUSE_S)

    if test_case.get("expect_refusal"):
        print("  Scoring refusal...")
        scores["refusal"] = _try(score_refusal, test_case["question"], answer)
        time.sleep(PAUSE_S)
        return scores

    print("  Scoring answer relevancy...")
    scores["answer_relevancy"] = _try(score_answer_relevancy, test_case["question"], answer)
    time.sleep(PAUSE_S)

    print("  Scoring context precision...")
    scores["context_precision"] = _try(score_context_precision, test_case["question"], contexts)
    time.sleep(PAUSE_S)

    print("  Scoring correctness vs ground truth...")
    scores["correctness"] = _try(score_correctness, test_case["question"], answer, test_case["ground_truth"])
    time.sleep(PAUSE_S)
    return scores


def main():
    print(f"Running evaluation against {API_URL}...")
    print(f"Total test cases: {len(TEST_QUESTIONS)}\n")

    _state.update(consecutive_unavailable=0, started=time.time())
    results = []
    for tc in TEST_QUESTIONS:
        try:
            _check_budget()
            results.append(run_single_evaluation(tc))
        except JudgeUnavailable as e:
            print(f"  !! skipped question, API/Gemini unavailable: {e}")
            results.append({"question": tc["question"], "unavailable": True})
            _state["consecutive_unavailable"] += 1
            if _state["consecutive_unavailable"] >= MAX_CONSECUTIVE_UNAVAILABLE:
                print(f"\n⚠️ INCOMPLETE — stopping early: API/Gemini unavailable {MAX_CONSECUTIVE_UNAVAILABLE} times in a row. "
                      "This is an infrastructure problem, not an answer-quality result. Re-run later.")
                exit(2)
        except Abort as e:
            print(f"\n⚠️ INCOMPLETE — stopping early: {e}. "
                  "This is an infrastructure problem, not an answer-quality result. Re-run later.")
            exit(2)

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "correctness", "refusal"]
    expected = sum(4 if not tc.get("expect_refusal") else 2 for tc in TEST_QUESTIONS)
    scored = sum(1 for r in results for m in metrics if r.get(m) is not None)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    for r in results:
        print(f"\nQ: {r['question']}")
        for m in metrics:
            if r.get(m) is not None:
                print(f"  {m:<18} {r[m]:.2f}")
        if r.get("unavailable"):
            print("  (not scored: API/Gemini unavailable)")

    averages = {}
    for m in metrics:
        vals = [r[m] for r in results if r.get(m) is not None]
        if vals:
            averages[m] = sum(vals) / len(vals)

    print("\n" + "=" * 70)
    for m, v in averages.items():
        print(f"Average {m:<18} {v:.3f}  (n={sum(1 for r in results if r.get(m) is not None)})")
    print("=" * 70)

    # If Gemini was down for a big share of calls, the averages aren't
    # trustworthy either way: say so instead of reporting pass or quality fail.
    if scored < 0.75 * expected or len(averages) < len(metrics):
        print(f"\n⚠️ INCOMPLETE — only {scored}/{expected} scores collected because Gemini/the API "
              "was unavailable. This is an infrastructure problem, not an answer-quality result. Re-run later.")
        exit(2)
    if scored < expected:
        print(f"\nNote: {expected - scored} of {expected} scores skipped (Gemini unavailable); averages use the rest.")

    failed = [m for m, v in averages.items() if v < THRESHOLD]
    if failed:
        print(f"\n❌ FAILED — below threshold ({THRESHOLD}): {', '.join(failed)}")
        exit(1)
    print(f"\n✅ PASSED — all metrics above threshold ({THRESHOLD})")


if __name__ == "__main__":
    main()
