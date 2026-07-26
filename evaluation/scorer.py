"""
Custom RAGAS-style evaluation scorer, built directly on Gemini —
no LangChain, no ragas library. This exists because ragas 0.3.9/0.4.3
both hit unresolvable dependency conflicts on this environment
(see git history / conversation notes for the full diagnosis).

Implements three focused checks, each scored 0.0-1.0 by asking
Gemini to judge the answer against specific criteria:

- faithfulness:      does the answer only use facts present in the
                      retrieved context (no invented information)?
- answer_relevancy:  does the answer actually address the question?
- context_precision: was the retrieved context actually useful for
                      answering this specific question?
"""

import os
import re
from google import genai

JUDGE_MODEL = "gemini-3.1-flash-lite"
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _extract_score(text: str) -> float:
    """
    Pulls a numeric score out of the judge's response.
    Expects the judge to output a line like 'SCORE: 0.8'.
    Falls back to 0.0 if no valid score is found, rather than
    crashing the whole evaluation run over one malformed response.
    """
    match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text)
    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))  # clamp to valid range
    return 0.0


def score_faithfulness(answer: str, contexts: list[str]) -> dict:
    """
    Judges whether the answer only contains information that's
    actually supported by the retrieved context — the core
    hallucination check.
    """
    context_text = "\n".join(f"- {c}" for c in contexts)
    prompt = f"""You are evaluating whether an AI-generated answer only uses facts
present in the provided context, with no invented information.

CONTEXT PROVIDED TO THE AI:
{context_text}

AI'S ANSWER:
{answer}

Does the answer ONLY contain claims that are supported by the context above?
Score 1.0 if fully faithful (no invented facts), 0.0 if it invents information
not present in the context, or something in between for partial issues.

Respond in exactly this format:
SCORE: <number between 0.0 and 1.0>
REASON: <one sentence explaining the score>"""

    client = _get_client()
    response = client.models.generate_content(model=JUDGE_MODEL, contents=prompt)
    return {"score": _extract_score(response.text), "raw": response.text}


def score_answer_relevancy(question: str, answer: str) -> dict:
    """
    Judges whether the answer actually addresses what was asked,
    rather than being generic, off-topic, or evasive.
    """
    prompt = f"""You are evaluating whether an AI's answer actually addresses
the user's question.

QUESTION:
{question}

ANSWER:
{answer}

Does the answer directly and specifically address the question asked?
Score 1.0 if fully relevant and on-topic, 0.0 if it doesn't address the
question at all, or something in between if only partially relevant.

Respond in exactly this format:
SCORE: <number between 0.0 and 1.0>
REASON: <one sentence explaining the score>"""

    client = _get_client()
    response = client.models.generate_content(model=JUDGE_MODEL, contents=prompt)
    return {"score": _extract_score(response.text), "raw": response.text}


def score_context_precision(question: str, contexts: list[str]) -> dict:
    """
    Judges whether the retrieved context was actually useful and
    relevant for answering this specific question — a retrieval
    quality check, independent of how the answer was generated.
    """
    context_text = "\n".join(f"- {c}" for c in contexts)
    prompt = f"""You are evaluating whether retrieved reference material is
actually useful for answering a specific question.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context_text}

Is this retrieved context genuinely relevant and useful for answering the
question above? Score 1.0 if highly relevant, 0.0 if irrelevant or unrelated,
or something in between if only partially useful.

Respond in exactly this format:
SCORE: <number between 0.0 and 1.0>
REASON: <one sentence explaining the score>"""

    client = _get_client()
    response = client.models.generate_content(model=JUDGE_MODEL, contents=prompt)
    return {"score": _extract_score(response.text), "raw": response.text}