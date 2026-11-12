"""
Tests for the chatbot evaluation harness (evaluation/). Gemini and the API are
faked throughout: these check the harness's own logic, not the chatbot.
"""

import json

import pytest

import run_eval
import scorer
from knowledge_base_mirror import get_texts_for_ids
from test_questions import TEST_QUESTIONS


def _reply(**scores):
    return json.dumps({m: {"reason": f"because {m}", "score": s} for m, s in scores.items()})


GOOD = _reply(faithfulness=1.0, answer_relevancy=0.9, context_precision=0.8, correctness=1.0)


# ── the test set itself ──────────────────────────────────────────────────────

def test_every_expected_context_id_exists_in_the_knowledge_base():
    for tc in TEST_QUESTIONS:
        ids = tc.get("expected_context", [])
        assert len(get_texts_for_ids(ids)) == len(ids), f"unknown knowledge-base id in: {tc['question']}"


# must match ASK_TOOL_FUNCTIONS in ml/serving/api.py (tests/api checks the API side)
KNOWN_TOOLS = {"get_forecast", "rank_categories", "rank_states", "get_forecast_accuracy"}


def test_test_set_is_well_formed():
    questions = [tc["question"] for tc in TEST_QUESTIONS]
    assert len(questions) == len(set(questions)), "duplicate question"
    for tc in TEST_QUESTIONS:
        assert tc["ground_truth"].strip()
        assert tc.get("lang", "en") in ("en", "ms")
        if tc.get("expect_refusal"):
            assert "expected_context" not in tc, "out-of-scope questions have no right entry to retrieve"
        elif tc.get("expected_tools"):
            assert set(tc["expected_tools"]) <= KNOWN_TOOLS, f"unknown tool in: {tc['question']}"
        else:
            assert tc["expected_context"], f"no expected_context for: {tc['question']}"
        if tc["needs_live_data"]:
            assert tc["state"] and tc["division"]


def test_test_set_covers_both_languages_and_refusals():
    assert sum(tc.get("lang") == "ms" for tc in TEST_QUESTIONS) >= 4
    assert sum(bool(tc.get("expect_refusal")) for tc in TEST_QUESTIONS) >= 3
    assert any(tc.get("lang") == "ms" and tc.get("expect_refusal") for tc in TEST_QUESTIONS)


def test_test_set_covers_every_tool_and_the_no_tool_case():
    data = [tc for tc in TEST_QUESTIONS if tc.get("expected_tools")]
    assert {name for tc in data for name in tc["expected_tools"]} == KNOWN_TOOLS
    assert any(tc.get("lang") == "ms" for tc in data)
    assert any(tc["needs_live_data"] for tc in data), "one must rely on what the user is viewing"
    assert sum(tc.get("expected_tools") == [] for tc in TEST_QUESTIONS) >= 5


# ── tool choice ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("called, expected, correct", [
    (["get_forecast"], ["get_forecast"], True),
    (["get_forecast", "get_forecast"], ["get_forecast", "rank_states"], True),   # any acceptable tool
    (["rank_categories"], ["get_forecast"], False),
    ([], ["get_forecast"], False),                                              # answered without data
    ([], [], True),                                                             # general question
    (["get_forecast"], [], False),                                              # wasted a call
])
def test_tool_choice_correct(called, expected, correct):
    assert run_eval.tool_choice_correct(called, expected) is correct


def test_data_questions_are_judged_against_what_the_tools_returned(monkeypatch):
    prompts = []
    monkeypatch.setattr(scorer, "_generate", lambda p: prompts.append(p) or GOOD)
    monkeypatch.setattr(run_eval, "call_ask_endpoint", lambda q, live, **kw: {
        "answer": "Food in Sabah is forecast to rise 0.6%.", "retrieved_context": ["data_sources"],
        "tools_called": ["get_forecast"],
        "tool_results": [{"name": "get_forecast", "args": {"state": "Sabah"},
                          "result": {"forecast": [{"month": "2026-12", "expected_change_pct": 0.6}]}}]})
    case = next(tc for tc in TEST_QUESTIONS if tc.get("expected_tools") == ["get_forecast"])
    record = run_eval.run_single_evaluation(case)

    assert record["tool_choice"]["correct"] is True
    assert set(record["scores"]) == {"faithfulness", "answer_relevancy"}    # no fixed reference answer
    assert 'DATA FROM TOOL get_forecast({"state": "Sabah"})' in prompts[0]
    assert '"expected_change_pct": 0.6' in prompts[0]
    assert '"correctness"' not in prompts[0]


def test_wrong_tool_choices_fail_the_run():
    results = [{"question": f"q{i}", "scores": {"faithfulness": 1.0, "answer_relevancy": 1.0,
                                                 "context_precision": 1.0, "correctness": 1.0, "refusal": 1.0},
                "tool_choice": {"correct": i < 2, "called": [], "expected": ["get_forecast"]}}
               for i in range(4)]
    summary = run_eval.summarise(results)
    assert summary["tools"] == {"accuracy": 0.5, "n": 4}
    status, reasons = run_eval.verdict(summary)
    assert status == "failed" and "tool choice accuracy 0.50" in reasons[0]


# ── judge reply parsing: never default a score ───────────────────────────────

def test_parse_judgement_reads_scores_and_reasons():
    out = scorer.parse_judgement(GOOD, scorer.ANSWER_METRICS)
    assert out["answer_relevancy"] == {"score": 0.9, "reason": "because answer_relevancy"}
    assert set(out) == set(scorer.ANSWER_METRICS)


def test_parse_judgement_tolerates_a_code_fence_and_bare_numbers():
    fenced = "```json\n" + json.dumps({"faithfulness": 1, "refusal": 0.5}) + "\n```"
    out = scorer.parse_judgement(fenced, scorer.REFUSAL_METRICS)
    assert out["faithfulness"]["score"] == 1.0 and out["refusal"]["score"] == 0.5


@pytest.mark.parametrize("bad", [
    "",                                                       # empty reply
    "SCORE: 0.8",                                             # the old text format
    "[1, 2]",                                                 # JSON, but not an object
    _reply(faithfulness=1.0, answer_relevancy=1.0, context_precision=1.0),          # a metric missing
    _reply(faithfulness=1.0, answer_relevancy=1.0, context_precision=1.0, correctness=7),   # out of range
    json.dumps({m: {"score": "high"} for m in scorer.ANSWER_METRICS}),              # not a number
    json.dumps({m: {"score": True} for m in scorer.ANSWER_METRICS}),                # a boolean is not a score
])
def test_parse_judgement_rejects_unusable_replies(bad):
    with pytest.raises(scorer.JudgeInvalid):
        scorer.parse_judgement(bad, scorer.ANSWER_METRICS)


def test_prompt_asks_only_for_the_metrics_that_apply():
    normal = scorer.build_prompt("Q?", "A.", ["ctx one"], "truth", expect_refusal=False)
    refusal = scorer.build_prompt("Q?", "A.", [], "truth", expect_refusal=True)
    assert all(f'"{m}"' in normal for m in scorer.ANSWER_METRICS) and '"refusal"' not in normal
    assert '"refusal"' in refusal and '"correctness"' not in refusal
    assert "ctx one" in normal and "(nothing was retrieved)" in refusal


def test_judge_makes_one_call_per_question(monkeypatch):
    calls = []
    monkeypatch.setattr(scorer, "_generate", lambda prompt: calls.append(prompt) or GOOD)
    out = scorer.judge("Q?", "A.", ["ctx"], "truth")
    assert len(calls) == 1
    assert out["correctness"]["score"] == 1.0


def test_judge_retries_once_then_reports_unscored(monkeypatch):
    replies = iter(["not json", GOOD])
    monkeypatch.setattr(scorer, "_generate", lambda prompt: next(replies))
    assert scorer.judge("Q?", "A.", ["ctx"], "truth")["faithfulness"]["score"] == 1.0

    monkeypatch.setattr(scorer, "_generate", lambda prompt: "still not json")
    with pytest.raises(scorer.JudgeInvalid):
        scorer.judge("Q?", "A.", ["ctx"], "truth")


# ── retrieval metrics (no LLM) ───────────────────────────────────────────────

@pytest.mark.parametrize("retrieved, expected, hit, rr", [
    (["a", "b", "c"], ["a"], True, 1.0),
    (["x", "a", "c"], ["a"], True, 0.5),
    (["x", "y", "a"], ["a", "b"], True, 1 / 3),
    (["x", "b", "a"], ["a", "b"], True, 0.5),          # the first expected entry found counts
    (["x", "y", "z"], ["a"], False, 0.0),
    ([], ["a"], False, 0.0),
])
def test_retrieval_scores(retrieved, expected, hit, rr):
    assert run_eval.retrieval_scores(retrieved, expected) == {"hit": hit, "reciprocal_rank": pytest.approx(rr)}


# ── summary and verdict ──────────────────────────────────────────────────────

def _case(correctness=1.0, hit=True, lang="en"):
    return {"question": f"q{correctness}{hit}{lang}", "lang": lang, "answer": "a",
            "retrieval": {"hit": hit, "reciprocal_rank": 1.0 if hit else 0.0},
            "scores": {"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0,
                       "correctness": correctness}}


def _refusal(score=1.0):
    return {"question": f"r{score}", "lang": "en", "answer": "a", "refusal_case": True,
            "scores": {"faithfulness": 1.0, "refusal": score}}


def _unscored():
    return {"question": "u", "lang": "en", "error": "unavailable: 503"}


def test_summarise_averages_only_what_was_scored():
    s = run_eval.summarise([_case(1.0), _case(0.5, hit=False, lang="ms"), _refusal(), _unscored()])
    assert s["questions"] == 4 and s["judged"] == 3 and s["asked"] == 3
    assert s["metrics"]["correctness"] == {"mean": 0.75, "n": 2}
    assert s["metrics"]["faithfulness"]["n"] == 3
    assert s["metrics"]["refusal"] == {"mean": 1.0, "n": 1}
    assert s["retrieval"] == {"hit_rate": 0.5, "mrr": 0.5, "n": 2}
    assert s["by_lang"] == {"en": {"correctness": 1.0, "n": 1}, "ms": {"correctness": 0.5, "n": 1}}


def test_verdict_passes_when_everything_clears_its_threshold():
    assert run_eval.verdict(run_eval.summarise([_case(), _case(0.8), _refusal()])) == ("passed", [])


def test_verdict_fails_on_a_low_judge_metric():
    status, reasons = run_eval.verdict(run_eval.summarise([_case(0.2), _case(0.4), _refusal()]))
    assert status == "failed" and "correctness" in reasons[0]


def test_verdict_fails_on_a_low_retrieval_hit_rate():
    status, reasons = run_eval.verdict(run_eval.summarise([_case(hit=False), _case(hit=False), _case(), _refusal()]))
    assert status == "failed" and "retrieval hit rate" in reasons[0]


def test_verdict_is_incomplete_not_failed_when_gemini_was_down():
    """An outage must never be reported as a quality failure (or a pass)."""
    status, _ = run_eval.verdict(run_eval.summarise([_case(0.1), _unscored(), _unscored(), _unscored()]))
    assert status == "incomplete"
    status, _ = run_eval.verdict(run_eval.summarise([_case(), _case()]))       # no refusal case scored
    assert status == "incomplete"
    assert run_eval.verdict(run_eval.summarise([]))[0] == "incomplete"


# ── the whole run, offline ───────────────────────────────────────────────────

@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Fake /ask (always retrieves the right entry) and the judge; no sleeping."""
    by_question = {tc["question"]: tc for tc in TEST_QUESTIONS}

    def fake_ask(question, needs_live_data, **kwargs):
        expected = by_question[question].get("expected_context", ["app_bilingual"])
        return {"answer": "An answer.", "retrieved_context": [expected[0], "data_sources"],
                "tools_called": by_question[question].get("expected_tools", [])[:1], "tool_results": []}

    def fake_generate(prompt):
        if '"refusal"' in prompt:
            return _reply(faithfulness=1.0, refusal=1.0)
        return GOOD

    monkeypatch.setattr(run_eval, "call_ask_endpoint", fake_ask)
    monkeypatch.setattr(scorer, "_generate", fake_generate)
    monkeypatch.setattr(run_eval, "PAUSE_S", 0)
    monkeypatch.setattr(run_eval, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    return tmp_path


def test_full_run_passes_and_writes_results(offline, capsys):
    assert run_eval.main() == 0

    report = json.loads((offline / "latest.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["summary"]["judged"] == len(TEST_QUESTIONS)
    assert report["summary"]["retrieval"]["hit_rate"] == 1.0
    assert report["summary"]["tools"]["accuracy"] == 1.0
    assert set(report["summary"]["by_lang"]) == {"en", "ms"}
    assert all("reasons" in r for r in report["results"])

    md = (offline / "latest.md").read_text(encoding="utf-8")
    assert "retrieval hit rate" in md and "passed" in md
    assert (offline / "summary.md").read_text(encoding="utf-8") == md       # CI job summary


def test_full_run_fails_when_answers_are_wrong(offline, monkeypatch):
    bad = _reply(faithfulness=1.0, answer_relevancy=1.0, context_precision=1.0, correctness=0.1)
    monkeypatch.setattr(scorer, "_generate",
                        lambda p: _reply(faithfulness=1.0, refusal=1.0) if '"refusal"' in p else bad)
    assert run_eval.main() == 1
    md = (offline / "latest.md").read_text(encoding="utf-8")
    assert "failed" in md and "Questions below threshold" in md


def test_full_run_is_incomplete_when_the_api_is_down(offline, monkeypatch):
    def down(*args, **kwargs):
        raise scorer.JudgeUnavailable("/ask kept failing: HTTP 503")
    monkeypatch.setattr(run_eval, "call_ask_endpoint", down)
    assert run_eval.main() == 2

    report = json.loads((offline / "latest.json").read_text(encoding="utf-8"))
    assert report["status"] == "incomplete"
    assert "stopped early" in report["reasons"][0]
    assert len(report["results"]) == len(TEST_QUESTIONS)         # unreached questions are listed too


def test_a_garbled_judge_reply_is_unscored_not_zero(offline, monkeypatch):
    first = TEST_QUESTIONS[0]["question"]
    monkeypatch.setattr(scorer, "_generate",
                        lambda p: "garbage" if first in p else
                        (_reply(faithfulness=1.0, refusal=1.0) if '"refusal"' in p else GOOD))
    assert run_eval.main() == 0                                   # one unscored question does not fail the run
    report = json.loads((offline / "latest.json").read_text(encoding="utf-8"))
    garbled = report["results"][0]
    assert "scores" not in garbled and "judge reply unusable" in garbled["error"]
    assert report["summary"]["judged"] == len(TEST_QUESTIONS) - 1
