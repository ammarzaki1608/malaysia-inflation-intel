"""
Evaluation runner for the /ask endpoint.

For every question in test_questions.py it calls the live /ask endpoint, then:

  1. Retrieval (no LLM): were the knowledge-base entries that hold the answer
     among the ones /ask retrieved?  -> hit rate and mean reciprocal rank (MRR)
  2. Answer quality (one Gemini judge call, see scorer.py): faithfulness,
     answer relevancy, context precision, correctness; or, for out-of-scope
     questions, faithfulness and refusal.

It prints a report, writes results/latest.json and results/latest.md, appends
the summary to the GitHub job summary when run in CI, and exits:

  0  passed      every judge metric >= THRESHOLD and hit rate >= RETRIEVAL_THRESHOLD
  1  failed      a metric is below its threshold (an answer-quality result)
  2  incomplete  Gemini or the API was unavailable too often (an infrastructure
                 problem, not a quality result: re-run later)

Usage:
    python run_eval.py

Requires:
    API_URL env var pointing at a running instance of the API
    GEMINI_API_KEY env var (used by the judge)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import httpx

from knowledge_base_mirror import get_texts_for_ids
from scorer import (ANSWER_METRICS, DATA_METRICS, JUDGE_MODEL, REFUSAL_METRICS, RETRY_WAITS_S,
                    JudgeInvalid, JudgeUnavailable, judge)
from test_questions import TEST_QUESTIONS

API_URL = os.getenv("API_URL", "http://localhost:8081")
# Marks these calls as evaluation traffic, so the API logs them (it logs nothing else).
EVAL_LOG_TOKEN = os.getenv("EVAL_LOG_TOKEN", "")
ASK_HEADERS = {"X-Eval-Token": EVAL_LOG_TOKEN} if EVAL_LOG_TOKEN else {}
THRESHOLD = 0.7                                                    # every judge metric's average
RETRIEVAL_THRESHOLD = float(os.getenv("EVAL_RETRIEVAL_THRESHOLD", "0.8"))   # retrieval hit rate
TOOL_THRESHOLD = float(os.getenv("EVAL_TOOL_THRESHOLD", "0.7"))             # tool choice accuracy
PAUSE_S = float(os.getenv("EVAL_PAUSE_SECONDS", "5"))              # stay under Gemini rate limits
# Stop early when Gemini is clearly down, instead of retrying every call:
MAX_CONSECUTIVE_UNAVAILABLE = int(os.getenv("EVAL_MAX_CONSECUTIVE_UNAVAILABLE", "3"))
MAX_MINUTES = float(os.getenv("EVAL_MAX_MINUTES", "30"))           # whole-run budget
MIN_SCORED_SHARE = 0.75                                            # below this the run is "incomplete"
RESULTS_DIR = os.getenv("EVAL_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

JUDGE_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "correctness", "refusal"]


class Abort(RuntimeError):
    """Gemini/API unavailable too often, or the time budget ran out."""


# ── pure helpers (unit-tested in tests/evaluation) ───────────────────────────

def retrieval_scores(retrieved_ids: list, expected_ids: list) -> dict:
    """
    Hit and reciprocal rank for one question. A hit means at least one expected
    entry was retrieved; the reciprocal rank is 1/position of the first one
    (1.0 if it came first, 0.5 if second, 0.0 if none was retrieved).
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_ids:
            return {"hit": True, "reciprocal_rank": 1.0 / rank}
    return {"hit": False, "reciprocal_rank": 0.0}


def tool_choice_correct(called: list, expected: list) -> bool:
    """
    Did the chatbot pick a sensible tool? Checked without any LLM.
    `expected` lists the acceptable tools: calling any one of them is correct.
    An empty list means the question needs no data, so calling nothing is correct.
    """
    if not expected:
        return not called
    return any(name in expected for name in called)


def tool_contexts(tool_results: list) -> list:
    """Tool results as text, so the judge can check the answer's numbers against them."""
    return [f"DATA FROM TOOL {t['name']}({json.dumps(t.get('args', {}), ensure_ascii=False)}): "
            f"{json.dumps(t.get('result'), ensure_ascii=False)}" for t in tool_results]


def summarise(results: list) -> dict:
    """Averages over the questions that were actually scored."""
    metrics = {}
    for m in JUDGE_METRICS:
        vals = [r["scores"][m] for r in results if m in r.get("scores", {})]
        if vals:
            metrics[m] = {"mean": sum(vals) / len(vals), "n": len(vals)}

    rs = [r["retrieval"] for r in results if r.get("retrieval")]
    retrieval = None
    if rs:
        retrieval = {"hit_rate": sum(x["hit"] for x in rs) / len(rs),
                     "mrr": sum(x["reciprocal_rank"] for x in rs) / len(rs),
                     "n": len(rs)}

    ts = [r["tool_choice"] for r in results if r.get("tool_choice")]
    tools = {"accuracy": sum(x["correct"] for x in ts) / len(ts), "n": len(ts)} if ts else None

    by_lang = {}
    for lang in sorted({r.get("lang", "en") for r in results}):
        sub = [r for r in results if r.get("lang", "en") == lang and "correctness" in r.get("scores", {})]
        if sub:
            by_lang[lang] = {"correctness": sum(r["scores"]["correctness"] for r in sub) / len(sub), "n": len(sub)}

    return {"metrics": metrics, "retrieval": retrieval, "tools": tools, "by_lang": by_lang,
            "questions": len(results),
            "judged": sum(1 for r in results if r.get("scores")),
            "asked": sum(1 for r in results if "answer" in r)}


def verdict(summary: dict) -> tuple:
    """
    ("passed" | "failed" | "incomplete", [reasons]).
    Incomplete wins over failed: if too few questions were scored, the
    averages are not trustworthy either way.
    """
    n = summary["questions"]
    if n == 0 or summary["judged"] < MIN_SCORED_SHARE * n:
        return "incomplete", [f"only {summary['judged']}/{n} questions were judged"]
    missing = [m for m in JUDGE_METRICS if m not in summary["metrics"]]
    if missing:
        return "incomplete", [f"no scores at all for: {', '.join(missing)}"]

    failed = [f"{m} {v['mean']:.2f} < {THRESHOLD}" for m, v in summary["metrics"].items() if v["mean"] < THRESHOLD]
    r = summary["retrieval"]
    if r and r["hit_rate"] < RETRIEVAL_THRESHOLD:
        failed.append(f"retrieval hit rate {r['hit_rate']:.2f} < {RETRIEVAL_THRESHOLD}")
    t = summary.get("tools")
    if t and t["accuracy"] < TOOL_THRESHOLD:
        failed.append(f"tool choice accuracy {t['accuracy']:.2f} < {TOOL_THRESHOLD}")
    return ("failed", failed) if failed else ("passed", [])


def to_markdown(report: dict) -> str:
    s, status = report["summary"], report["status"]
    icon = {"passed": "✅", "failed": "❌", "incomplete": "⚠️"}[status]
    lines = [f"## Chatbot evaluation: {icon} {status}", "",
             f"{s['judged']} of {s['questions']} questions judged · judge `{report['judge_model']}` · "
             f"{report['duration_s']:.0f}s · {report['generated_at'][:16].replace('T', ' ')} UTC", ""]
    if report["reasons"]:
        lines += ["**Why:** " + "; ".join(report["reasons"]), ""]

    lines += ["| Metric | Score | Questions | Threshold |", "|---|---:|---:|---:|"]
    for m in JUDGE_METRICS:
        if m in s["metrics"]:
            v = s["metrics"][m]
            lines.append(f"| {m.replace('_', ' ')} | {v['mean']:.2f} | {v['n']} | {THRESHOLD} |")
    if s["retrieval"]:
        r = s["retrieval"]
        lines.append(f"| retrieval hit rate (no LLM) | {r['hit_rate']:.2f} | {r['n']} | {RETRIEVAL_THRESHOLD} |")
        lines.append(f"| retrieval MRR (no LLM) | {r['mrr']:.2f} | {r['n']} | — |")
    if s.get("tools"):
        t = s["tools"]
        lines.append(f"| tool choice accuracy (no LLM) | {t['accuracy']:.2f} | {t['n']} | {TOOL_THRESHOLD} |")
    if len(s["by_lang"]) > 1:
        lines += ["", "Correctness by language: " + ", ".join(
            f"{lang} {v['correctness']:.2f} (n={v['n']})" for lang, v in s["by_lang"].items())]

    low = [r for r in report["results"] if any(v < THRESHOLD for v in r.get("scores", {}).values())
           or (r.get("retrieval") and not r["retrieval"]["hit"])
           or (r.get("tool_choice") and not r["tool_choice"]["correct"])]
    if low:
        lines += ["", "### Questions below threshold", ""]
        for r in low:
            bits = [f"{m} {v:.2f}" for m, v in r.get("scores", {}).items() if v < THRESHOLD]
            if r.get("retrieval") and not r["retrieval"]["hit"]:
                bits.append("retrieval miss")
            if r.get("tool_choice") and not r["tool_choice"]["correct"]:
                tc = r["tool_choice"]
                bits.append(f"tool choice: called {tc['called'] or 'nothing'}, "
                            f"expected {' or '.join(tc['expected']) or 'nothing'}")
            lines.append(f"- **{r['question']}** — {', '.join(bits)}")
    unscored = [r for r in report["results"] if not r.get("scores")]
    if unscored:
        lines += ["", "### Not scored", ""] + [f"- {r['question']} — {r.get('error', 'unknown')}" for r in unscored]
    return "\n".join(lines) + "\n"


# ── network ──────────────────────────────────────────────────────────────────

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
            response = httpx.post(f"{API_URL}/ask", json=payload, headers=ASK_HEADERS, timeout=60.0)
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


def run_single_evaluation(test_case: dict) -> dict:
    """Ask one question and score it. Raises JudgeUnavailable if /ask stays down."""
    print(f"\nAsking: {test_case['question']}")
    refusal = bool(test_case.get("expect_refusal"))
    record = {"question": test_case["question"], "lang": test_case.get("lang", "en"), "refusal_case": refusal}

    result = call_ask_endpoint(
        test_case["question"],
        test_case["needs_live_data"],
        state=test_case.get("state"),
        division=test_case.get("division"),
        horizon_months=test_case.get("horizon_months", 6),
    )
    record["answer"] = result["answer"]
    record["retrieved_context"] = result["retrieved_context"]

    expected = test_case.get("expected_context")
    if expected:
        record["expected_context"] = expected
        record["retrieval"] = retrieval_scores(result["retrieved_context"], expected)
        r = record["retrieval"]
        print(f"  retrieval: {'hit' if r['hit'] else 'MISS'} (reciprocal rank {r['reciprocal_rank']:.2f})")

    called = result.get("tools_called", [])
    record["tools_called"] = called
    if "expected_tools" in test_case:
        record["tool_choice"] = {"correct": tool_choice_correct(called, test_case["expected_tools"]),
                                 "called": called, "expected": test_case["expected_tools"]}
        print(f"  tool choice: {'ok' if record['tool_choice']['correct'] else 'WRONG'} "
              f"(called {called or 'nothing'})")

    # A question that needs forecast data has no fixed right answer, so it is
    # judged on whether the answer matches what the tools returned.
    data_question = bool(test_case.get("expected_tools"))
    contexts = get_texts_for_ids(result["retrieved_context"]) + tool_contexts(result.get("tool_results", []))
    try:
        judged = judge(test_case["question"], result["answer"], contexts,
                       test_case["ground_truth"], expect_refusal=refusal,
                       metrics=DATA_METRICS if data_question else None)
        record["scores"] = {m: v["score"] for m, v in judged.items()}
        record["reasons"] = {m: v["reason"] for m, v in judged.items()}
        print("  " + "  ".join(f"{m} {s:.2f}" for m, s in record["scores"].items()))
    except JudgeInvalid as e:
        record["error"] = f"judge reply unusable: {e}"
        print(f"  !! not scored: {record['error']}")
    return record


def run_all(test_cases: list) -> list:
    """Every question, stopping early (Abort) if Gemini/the API is clearly down."""
    started, consecutive, results = time.time(), 0, []
    for tc in test_cases:
        if (time.time() - started) / 60 > MAX_MINUTES:
            raise Abort(f"time budget of {MAX_MINUTES:.0f} min exceeded", results)
        try:
            results.append(run_single_evaluation(tc))
            consecutive = 0
        except JudgeUnavailable as e:
            print(f"  !! skipped question, API/Gemini unavailable: {e}")
            results.append({"question": tc["question"], "lang": tc.get("lang", "en"),
                            "refusal_case": bool(tc.get("expect_refusal")), "error": f"unavailable: {e}"})
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_UNAVAILABLE:
                raise Abort(f"API/Gemini unavailable {consecutive} times in a row", results)
        time.sleep(PAUSE_S)
    return results


def write_outputs(report: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    md = to_markdown(report)
    with open(os.path.join(RESULTS_DIR, "latest.md"), "w", encoding="utf-8") as f:
        f.write(md)
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write(md)
    print(f"\nWrote {RESULTS_DIR}/latest.json and latest.md")


def main() -> int:
    print(f"Running evaluation against {API_URL} (judge: {JUDGE_MODEL})")
    print(f"Total test cases: {len(TEST_QUESTIONS)}")

    started, aborted = time.time(), None
    try:
        results = run_all(TEST_QUESTIONS)
    except Abort as e:
        aborted, results = e.args[0], e.args[1]
        # questions never reached still count as not scored
        done = {r["question"] for r in results}
        results += [{"question": tc["question"], "lang": tc.get("lang", "en"),
                     "refusal_case": bool(tc.get("expect_refusal")), "error": "not reached"}
                    for tc in TEST_QUESTIONS if tc["question"] not in done]

    summary = summarise(results)
    status, reasons = verdict(summary)
    if aborted:
        status, reasons = "incomplete", [f"stopped early: {aborted}"] + (reasons if status == "incomplete" else [])

    report = {
        "status": status, "reasons": reasons,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": time.time() - started,
        "api_url": API_URL, "judge_model": JUDGE_MODEL,
        "thresholds": {"judge_metrics": THRESHOLD, "retrieval_hit_rate": RETRIEVAL_THRESHOLD,
                       "tool_choice_accuracy": TOOL_THRESHOLD},
        "metrics_per_case": {"answer": list(ANSWER_METRICS), "refusal": list(REFUSAL_METRICS),
                             "data": list(DATA_METRICS)},
        "summary": summary, "results": results,
    }
    write_outputs(report)

    print("\n" + "=" * 70)
    print(to_markdown(report))
    print("=" * 70)
    if status == "incomplete":
        print(f"\n⚠️ INCOMPLETE — {'; '.join(reasons)}. "
              "This is an infrastructure problem, not an answer-quality result. Re-run later.")
        return 2
    if status == "failed":
        print(f"\n❌ FAILED — {'; '.join(reasons)}")
        return 1
    print(f"\n✅ PASSED — all metrics at or above threshold "
          f"({THRESHOLD}; retrieval {RETRIEVAL_THRESHOLD}; tool choice {TOOL_THRESHOLD})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
