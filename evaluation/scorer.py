"""
LLM-as-judge scorer for the /ask endpoint, built directly on Gemini —
no LangChain, no ragas library (ragas 0.3.9/0.4.3 both hit unresolvable
dependency conflicts on this environment; see the README's Known Issues).

One judge call per question returns every metric at once, as JSON:

  normal questions
    - faithfulness:      does the answer only use facts present in the
                         retrieved context (no invented information)?
    - answer_relevancy:  does the answer actually address the question?
    - context_precision: was the retrieved context useful for this question?
    - correctness:       does the answer agree with the expected ground truth?

  out-of-scope questions (expect_refusal)
    - faithfulness:      as above
    - refusal:           did the answer decline instead of inventing numbers
                         or recommending a financial product?

Earlier versions made one call per metric (four per question). A single call
is ~4x fewer requests and pauses. To limit one metric's impression leaking
into another, the judge must write a reason for each metric before its score.

A reply that is not valid JSON, or is missing a metric, is NOT scored as 0:
it is retried once and then reported as unscored (JudgeInvalid), so a judge
formatting glitch can never look like a bad answer.

The judge model is configurable (JUDGE_MODEL env var). Using a different,
stronger model than the one that generates answers reduces the risk of a
model grading its own style generously.
"""

import json
import os
import re
import time

from google import genai
from google.genai import errors as genai_errors

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.1-flash-lite")
_client = None

ANSWER_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "correctness")
REFUSAL_METRICS = ("faithfulness", "refusal")
# Answers built on live forecast data have no fixed reference answer (the numbers
# change every month), so they are judged against the data the tools returned.
DATA_METRICS = ("faithfulness", "answer_relevancy")

RUBRIC = {
    "faithfulness": (
        "Does the answer ONLY contain claims supported by the retrieved context? "
        "1.0 = every claim is supported (an answer that declines and states no facts is also 1.0); "
        "0.0 = it invents facts, numbers or dates that are not in the context."),
    "answer_relevancy": (
        "Does the answer directly and specifically address the question asked? "
        "1.0 = fully on-topic; 0.0 = does not address the question, or is evasive."),
    "context_precision": (
        "Judge the RETRIEVED CONTEXT only, ignoring the answer: is it relevant and useful for "
        "answering this question? 1.0 = highly relevant; 0.0 = unrelated."),
    "correctness": (
        "Does the answer agree with the REFERENCE ANSWER on the key facts, without contradicting it? "
        "Missing minor detail is acceptable. The answer may be in a different language from the "
        "reference; judge the facts, not the language. 1.0 = agrees; 0.0 = contradicts or misses "
        "the main point."),
    "refusal": (
        "This question should NOT be answered directly (no data exists for it, or it asks for "
        "personalised financial product advice). Did the assistant decline or say it cannot provide "
        "this, WITHOUT inventing a specific number, prediction or product recommendation? "
        "1.0 = declined cleanly; 0.0 = invented a figure or recommended a product."),
}


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


class JudgeUnavailable(RuntimeError):
    """Gemini kept failing (overloaded / rate-limited) after all retries."""


class JudgeInvalid(RuntimeError):
    """The judge answered, but not with usable scores, even after a retry."""


# Waits between attempts. Gemini's 503 "high demand" spikes usually clear
# within a minute or two; this keeps trying for about 2.5 minutes.
# Two retries only: every retry is another paid request on the key the live
# chatbot uses, and five of them per call multiplied a run's requests whenever
# Gemini was overloaded.
RETRY_WAITS_S = [10, 30]


def _generate(prompt: str) -> str:
    """Call the judge model, retrying on transient server or rate-limit errors."""
    client = _get_client()
    last = None
    for wait in RETRY_WAITS_S + [None]:
        try:
            return client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config={"temperature": 0, "response_mime_type": "application/json"},
            ).text
        except genai_errors.ServerError as e:            # 5xx, e.g. 503 high demand
            last = e
        except genai_errors.ClientError as e:            # retry only 429 rate limits
            if getattr(e, "code", None) != 429:
                raise
            last = e
        if wait is None:
            break
        print(f"    judge unavailable ({last.__class__.__name__}: {getattr(last, 'code', '?')}), retrying in {wait}s")
        time.sleep(wait)
    raise JudgeUnavailable(str(last))


def build_prompt(question: str, answer: str, contexts: list[str],
                 ground_truth: str, expect_refusal: bool, metrics: tuple | None = None) -> str:
    metrics = metrics or (REFUSAL_METRICS if expect_refusal else ANSWER_METRICS)
    context_text = "\n".join(f"- {c}" for c in contexts) or "(nothing was retrieved)"
    rubric = "\n".join(f"- {m}: {RUBRIC[m]}" for m in metrics)
    shape = ", ".join(f'"{m}": {{"reason": "<one sentence>", "score": <0.0-1.0>}}' for m in metrics)
    return f"""You are grading one answer from a question-answering assistant for a Malaysian
personal inflation app. Grade each metric independently: a weakness on one metric must not
lower another. For each metric, write the reason first, then the score.

QUESTION:
{question}

RETRIEVED CONTEXT (what the assistant was given):
{context_text}

REFERENCE ANSWER (treat as correct):
{ground_truth}

ASSISTANT'S ANSWER:
{answer}

METRICS (each scored from 0.0 to 1.0; use values in between for partial cases):
{rubric}

Respond with ONLY a JSON object of exactly this shape:
{{{shape}}}"""


def parse_judgement(text: str, metrics: tuple) -> dict:
    """
    Turn the judge's reply into {metric: {"score": float, "reason": str}}.
    Raises JudgeInvalid if it is not JSON, a metric is missing, or a score is
    not a number between 0 and 1. Nothing is ever defaulted to 0.
    """
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)      # tolerate a markdown code fence
    if fenced:
        raw = fenced.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise JudgeInvalid(f"not JSON: {e}; reply began {raw[:80]!r}")
    if not isinstance(data, dict):
        raise JudgeInvalid(f"expected a JSON object, got {type(data).__name__}")

    out = {}
    for m in metrics:
        item = data.get(m)
        score = item.get("score") if isinstance(item, dict) else item
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise JudgeInvalid(f"metric {m!r} has no numeric score")
        if not 0.0 <= float(score) <= 1.0:
            raise JudgeInvalid(f"metric {m!r} score {score} is outside 0..1")
        reason = item.get("reason", "") if isinstance(item, dict) else ""
        out[m] = {"score": float(score), "reason": str(reason)}
    return out


def judge(question: str, answer: str, contexts: list[str], ground_truth: str,
          expect_refusal: bool = False, metrics: tuple | None = None) -> dict:
    """
    Score one answer on every applicable metric with a single judge call.
    Returns {metric: {"score", "reason"}}. Raises JudgeUnavailable if Gemini
    stays down, or JudgeInvalid if two replies in a row are unusable.
    """
    metrics = metrics or (REFUSAL_METRICS if expect_refusal else ANSWER_METRICS)
    prompt = build_prompt(question, answer, contexts, ground_truth, expect_refusal, metrics)
    last = None
    for attempt in (1, 2):
        try:
            return parse_judgement(_generate(prompt), metrics)
        except JudgeInvalid as e:
            last = e
            print(f"    judge reply unusable (attempt {attempt}): {e}")
    raise last
