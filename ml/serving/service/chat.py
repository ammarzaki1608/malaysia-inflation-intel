"""The two endpoints that call Gemini: /ask (the chatbot) and /prescriptive-insight."""

import os
import sys
import time

from fastapi import APIRouter, HTTPException, Request
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel

from service import forecasts, llm_log
from service.chat_tools import ASK_TOOLS, horizon, run_tool
from service.config import CATEGORIES, STATES, limiter, logger

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag"))
from retriever import load_index_from_gcs, retrieve  # noqa: E402, F401  (load_index_from_gcs: used by api.py)

router = APIRouter()

GENERATION_MODEL = "gemini-3.1-flash-lite"
ASK_MAX_TOOL_ROUNDS = 2      # model turns that may request tools, before a final answer is forced
ASK_MAX_TOOL_CALLS = 6       # tool calls per question, across all rounds
LLM_UNAVAILABLE = "The assistant is temporarily unavailable. Please try again in a moment."

_genai_client = None


def genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client()
    return _genai_client


# ── /ask ─────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    state: str | None = None
    division: str | None = None
    horizon_months: int = 6


ASK_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant for a Malaysian personal inflation forecasting app. "
    "You must ONLY use the domain knowledge provided and the data returned by your tools — never invent "
    "numbers, dates, or facts that are not in them. If the question cannot be answered from them, say "
    "so honestly rather than guessing. Keep answers concise, in plain language, and answer in the "
    "language the question was asked in.\n"
    "Tools: call a tool when the question needs forecast numbers, a comparison between states or "
    "categories, or the forecasts' track record. Do not call a tool for questions about how the app "
    "works or what a term means; answer those from the domain knowledge alone. When the user says "
    "'my' or 'this' forecast or category, use the state and category they are viewing. The forecasts "
    "cover one state and one category at a time: there is no national forecast, so do not present a "
    "state figure as a national one. Never recommend specific financial products."
)


def build_ask_prompt(body: AskRequest, retrieved: list) -> str:
    """
    The first message to the model: the retrieved knowledge, what the user has
    open in the app, and the question. What the user has open is only a hint;
    whether to fetch numbers for it (or for anything else) is the model's decision.
    """
    knowledge_context = "\n\n".join([f"- {r['text']}" for r in retrieved])

    viewing = "Nothing selected."
    if body.state and body.division:
        viewing = (f"State: {body.state}. Category: {CATEGORIES[body.division]}. "
                   f"Forecast horizon: {horizon(body.horizon_months)} months.")
    elif body.state:
        viewing = f"State: {body.state}."

    return f"""DOMAIN KNOWLEDGE:
{knowledge_context}

WHAT THE USER IS VIEWING IN THE APP:
{viewing}

USER QUESTION:
{body.question}

Answer the user's question using only the domain knowledge above and data from your tools."""


def answer_with_tools(prompt: str, tool_calls: list) -> dict:
    """
    Let the model call tools, then answer. Each tool call is appended to
    `tool_calls` as it happens, so the caller still has them if a later model
    call fails.

    The loop is capped: after ASK_MAX_TOOL_ROUNDS rounds of tool calls, or
    ASK_MAX_TOOL_CALLS calls in total, the model may no longer ask for tools
    and must answer.
    """
    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    usage: dict = {}
    generation_ms, llm_calls, answer, response = 0.0, 0, None, None

    for round_no in range(ASK_MAX_TOOL_ROUNDS + 1):
        final_round = round_no == ASK_MAX_TOOL_ROUNDS or len(tool_calls) >= ASK_MAX_TOOL_CALLS
        call_started = time.perf_counter()
        response = genai_client().models.generate_content(
            model=GENERATION_MODEL,
            contents=contents,
            config={
                "system_instruction": ASK_SYSTEM_INSTRUCTION,
                "max_output_tokens": 400,
                "temperature": 0.3,
                "tools": ASK_TOOLS,
                "automatic_function_calling": {"disable": True},     # this loop runs the tools
                # on the final round the model may no longer ask for tools
                "tool_config": {"function_calling_config": {"mode": "NONE" if final_round else "AUTO"}},
            },
        )
        generation_ms += (time.perf_counter() - call_started) * 1000
        llm_calls += 1
        llm_log.add_usage(usage, response)

        requested = list(getattr(response, "function_calls", None) or [])
        if not requested or final_round:
            answer = response.text
            break

        # Send the model's own turn back unchanged (it carries the signature
        # Gemini needs to continue), followed by each tool's result.
        contents.append(response.candidates[0].content)
        result_parts = []
        for call in requested[:ASK_MAX_TOOL_CALLS - len(tool_calls)]:
            args = dict(call.args or {})
            result = run_tool(call.name, args)
            tool_calls.append({"name": call.name, "args": args, "result": result})
            result_parts.append(genai_types.Part(function_response=genai_types.FunctionResponse(
                name=call.name, response=result, id=getattr(call, "id", None))))
        contents.append(genai_types.Content(role="user", parts=result_parts))

    return {"answer": answer, "usage": usage, "llm_calls": llm_calls,
            "generation_ms": round(generation_ms), "finish_reason": llm_log.finish_reason(response)}


@router.post("/ask")
@limiter.limit("15/minute")
async def ask(request: Request, body: AskRequest):
    """
    Answers a question about the app or about the forecasts.

    App knowledge is retrieved for every question. The model can then call
    tools (see chat_tools.py) to fetch forecast data it needs, for any state
    or category, not only the one open in the app.

    Calls from the evaluation run are recorded (llm_log.log_eval_call); users'
    calls are not. If retrieval or the model fails (for example Gemini is
    overloaded), the client gets a 503 it can retry, not an unexplained 500.
    """
    # reject bad input before spending any LLM calls
    if body.state is not None and body.state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown state: {body.state}")
    if body.division is not None and body.division not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown division code: {body.division}")

    started = time.perf_counter()
    log = {
        "endpoint": "/ask", "model": GENERATION_MODEL, "status": "ok",
        "question": body.question, "question_chars": len(body.question),
        "state": body.state, "division": body.division, "horizon_months": body.horizon_months,
        "used_live_forecast_data": False,
        "forecast_version": forecasts.forecast_meta.get("version"),
    }
    tool_calls: list[dict] = []

    try:
        retrieved = retrieve(body.question, top_k=3)
        log["retrieval_ms"] = round((time.perf_counter() - started) * 1000)
        log["retrieved_ids"] = [r["id"] for r in retrieved]
        log["retrieved_distances"] = [round(float(r["distance"]), 4) for r in retrieved if "distance" in r]

        prompt = build_ask_prompt(body, retrieved)
        outcome = answer_with_tools(prompt, tool_calls)
        answer = outcome["answer"]

        log["generation_ms"] = outcome["generation_ms"]
        log["llm_calls"] = outcome["llm_calls"]
        log["tools_called"] = [c["name"] for c in tool_calls]
        log["tool_errors"] = sum(1 for c in tool_calls if "error" in c["result"])
        log["used_live_forecast_data"] = any("error" not in c["result"] for c in tool_calls)
        log["prompt_chars"] = len(prompt)
        log["finish_reason"] = outcome["finish_reason"]
        log.update(outcome["usage"])
        if not answer:
            raise RuntimeError("the model returned no answer text")
        log["answer"] = answer
        log["answer_chars"] = len(answer)
    except Exception as e:
        log.update(status="error", error_type=e.__class__.__name__, error_message=str(e)[:500],
                   total_ms=round((time.perf_counter() - started) * 1000))
        llm_log.log_eval_call(request, log)
        logger.error(f"/ask failed: {e.__class__.__name__}: {e}")
        raise HTTPException(status_code=503, detail=LLM_UNAVAILABLE)

    log["total_ms"] = round((time.perf_counter() - started) * 1000)
    llm_log.log_eval_call(request, log)

    return {
        "question": body.question,
        "answer": answer,
        "retrieved_context": [r["id"] for r in retrieved],
        "used_live_forecast_data": log["used_live_forecast_data"],
        "tools_called": log["tools_called"],
        # what each tool returned, so an answer's numbers can be checked against them
        "tool_results": tool_calls,
    }


# ── /prescriptive-insight ────────────────────────────────────────────────────

class PrescriptiveRequest(BaseModel):
    context_type: str  # "personal_calculator" or "relocation_planner"
    summary_data: dict


INSIGHT_SYSTEM_INSTRUCTION = (
    "You are a budgeting assistant for a Malaysian personal inflation app. "
    "Generate ONE short paragraph (2-3 sentences) offering a general, actionable "
    "budgeting suggestion based on the data provided. "
    "STRICT RULES: "
    "1. Never recommend specific financial products, insurers, banks, or investments. "
    "2. Never use directive language like 'you should switch to X' or 'invest in Y'. "
    "3. Only suggest general actions like reviewing a specific expense category, "
    "building a buffer, or planning ahead — never specific financial products. "
    "4. Stay grounded only in the numbers provided — never invent figures. "
    "5. Keep it concise and practical, not alarmist."
)


def build_insight_prompt(body: PrescriptiveRequest) -> str:
    d = body.summary_data
    if body.context_type == "personal_calculator":
        return f"""REPORT DATA:
Biggest leak category: {d.get('biggest_leak_name')} (rising {d.get('biggest_leak_pct')}%)
Best category: {d.get('biggest_saving_name')}
Current monthly surplus: RM{d.get('current_surplus')}
Projected monthly surplus: RM{d.get('projected_surplus')}
Horizon: {d.get('horizon_months')} months

Generate a short, general budgeting suggestion based on this report."""
    return f"""RELOCATION COMPARISON DATA:
Origin state: {d.get('origin_state')}
Destination state: {d.get('destination_state')}
Monthly cost difference: RM{d.get('cost_diff_rm')}
Recommended minimum salary: RM{d.get('salary_needed')}
Horizon: {d.get('horizon_months')} months

Generate a short, general suggestion for someone considering this move."""


@router.post("/prescriptive-insight")
@limiter.limit("15/minute")
async def prescriptive_insight(request: Request, body: PrescriptiveRequest):
    """
    Generates a short, action-oriented suggestion grounded in the
    user's own report numbers. Deliberately stays in general-budgeting
    territory — never recommends specific financial products, insurers,
    or investments, since this app is not a licensed financial advisor.
    """
    prompt = build_insight_prompt(body)

    # Only the shape of the call is recorded, never the user's figures.
    started = time.perf_counter()
    log = {"endpoint": "/prescriptive-insight", "model": GENERATION_MODEL, "status": "ok",
           "context_type": body.context_type, "prompt_chars": len(prompt)}
    try:
        response = genai_client().models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config={
                "system_instruction": INSIGHT_SYSTEM_INSTRUCTION,
                "max_output_tokens": 150,
                "temperature": 0.4,
            },
        )
        insight = response.text
        log["generation_ms"] = round((time.perf_counter() - started) * 1000)
        log["answer_chars"] = len(insight or "")
        log["finish_reason"] = llm_log.finish_reason(response)
        log.update(llm_log.token_usage(response))
    except Exception as e:
        log.update(status="error", error_type=e.__class__.__name__, error_message=str(e)[:500],
                   total_ms=round((time.perf_counter() - started) * 1000))
        llm_log.log_eval_call(request, log)
        logger.error(f"/prescriptive-insight failed: {e.__class__.__name__}: {e}")
        raise HTTPException(status_code=503, detail=LLM_UNAVAILABLE)

    log["total_ms"] = round((time.perf_counter() - started) * 1000)
    llm_log.log_eval_call(request, log)

    return {"insight": insight}
