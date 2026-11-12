"""
Logging of LLM calls, for our own evaluation traffic only.

Calls from the app's users are not recorded anywhere: what people type into
the chat is theirs. The evaluation harness sends a secret token in the
x-eval-token header, and only those calls are written.
"""

import hmac
import json
import os
import uuid
from datetime import datetime, timezone

from fastapi import Request

from service import data
from service.config import API_REVISION, PROJECT_ID, logger

# One row per logged call. The table is created by Terraform.
LLM_LOG_TABLE = os.getenv("LLM_LOG_TABLE", f"{PROJECT_ID}.observability.llm_calls")
LLM_LOG_ENABLED = os.getenv("LLM_LOG_ENABLED", "true").lower() == "true"
# Only calls carrying this token (the evaluation run) are logged. Empty = log nothing.
EVAL_LOG_TOKEN = os.getenv("EVAL_LOG_TOKEN", "")
EVAL_TOKEN_HEADER = "x-eval-token"


def token_usage(response) -> dict:
    """Token counts from a Gemini response; empty if the SDK did not report them."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    }


def add_usage(total: dict, response) -> None:
    """Add one response's token counts to a running total."""
    for key, value in token_usage(response).items():
        if value is not None:
            total[key] = total.get(key, 0) + value


def finish_reason(response):
    try:
        reason = response.candidates[0].finish_reason
        return getattr(reason, "name", None) or (str(reason) if reason is not None else None)
    except Exception:
        return None


def log_llm_call(record: dict) -> None:
    """
    Record one LLM call: what was asked, what was retrieved, how long each stage
    took, how many tokens it used and whether it worked.

    Only ever called for our own evaluation traffic (see log_eval_call), so the
    question and answer stored here are test questions, never a user's.

    Two destinations:
      - a structured log line (metadata only, no question or answer text), so
        latency and errors are visible in Cloud Logging straight away;
      - one row in BigQuery (LLM_LOG_TABLE) including the question and answer,
        for analysis: slow calls, token spend, weak retrievals. Rows expire
        after 90 days.

    It runs before the response is returned, not as a background task: Cloud
    Run throttles CPU once a response has been sent, so work scheduled after it
    may never finish. It must never break the request, so every failure here
    is logged and swallowed.
    """
    row = {
        "call_id": uuid.uuid4().hex,
        "ts": datetime.now(timezone.utc).isoformat(),
        "api_revision": API_REVISION,
        **record,
    }
    try:
        meta = {k: v for k, v in row.items() if k not in ("question", "answer")}
        logger.info("llm_call " + json.dumps(meta, default=str))
    except Exception as e:                                   # pragma: no cover - defensive
        logger.warning(f"Could not write llm_call log line: {e}")

    if not LLM_LOG_ENABLED:
        return
    try:
        errors = data.client().insert_rows_json(LLM_LOG_TABLE, [row], timeout=5, ignore_unknown_values=True)
        if errors:
            logger.warning(f"LLM log row rejected by BigQuery: {errors}")
    except Exception as e:
        logger.warning(f"LLM log not written ({e.__class__.__name__}: {e})")


def is_eval_call(request: Request) -> bool:
    """True only when the caller sent our evaluation token."""
    sent = request.headers.get(EVAL_TOKEN_HEADER, "")
    if not EVAL_LOG_TOKEN or not sent:
        return False
    return hmac.compare_digest(sent.encode(), EVAL_LOG_TOKEN.encode())


def log_eval_call(request: Request, record: dict) -> None:
    """Log the call only if it came from our own evaluation run."""
    if is_eval_call(request):
        log_llm_call(record)
