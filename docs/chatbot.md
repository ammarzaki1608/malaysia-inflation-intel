# The chatbot: retrieval, tools, evaluation and logging

How `/ask` works, how it is tested, and what is (and is not) recorded. The short version is in the
[README](../README.md#the-chatbot).

## How it answers

```
Knowledge base (27 hand-curated entries)
        │  embedded once via gemini-embedding-001 (RETRIEVAL_DOCUMENT)
        ▼
FAISS flat L2 index  ←── stored in GCS, loaded once at API startup
        │
        │  live question ──► embedded via gemini-embedding-001 (RETRIEVAL_QUERY)
        ▼
Top-k retrieval  (always)
        ▼
gemini-3.1-flash-lite  ←── temperature 0.3, strict "never invent facts" system instruction
        │  ▲
        │  │  tool calls, when the question needs numbers (at most 2 rounds):
        ▼  │  get_forecast · rank_categories · rank_states · get_forecast_accuracy
Forecast table and scorecard already in memory
        ▼
Grounded answer, with the retrieved entries and every tool result returned for traceability
```

**Tool calling.** The model decides for itself whether a question needs data. It is given four tools (plain Python functions in `service/chat_tools.py`) and replies either with an answer or with a request such as `get_forecast(state="Kedah", category="Transport", horizon_months=6)`. The API runs the function, sends the result back, and the model writes its answer from it.

This replaced a keyword list that guessed whether a question was about "my forecast", and could only ever supply the one forecast open in the app. The chatbot can now answer questions the old version could not: "Is transport rising faster in Selangor or Johor?", "Which category will rise most in Kedah?", "How have the forecasts performed so far?".

| Decision | Reason |
|---|---|
| Plain Gemini function calling, no agent framework | Under 300 lines including the tools, no new dependency, every step visible in one function |
| Retrieval stays automatic; only data is a tool | A general question still costs one model call, as before. Only questions that need numbers pay for a second |
| Tools read the forecast table in memory | A tool call takes milliseconds and no database query |
| Hard limits: 2 rounds, 6 tool calls | On the last round the model is not allowed to ask for tools, so the loop always ends |
| Tools never raise | A bad argument comes back to the model as an error message listing the valid values, so it can correct itself or tell the user |
| Every argument is validated | States and categories are checked against the known lists before anything is looked up |

The cost: a question that needs data now takes two model calls, so roughly twice the time and tokens of a general question.

**Evaluation:** a custom Gemini-based LLM-as-judge harness (built after the standard `ragas` library hit an unresolvable dependency conflict — see [Known Issues](../README.md#known-issues--engineering-notes)) scores answers against a versioned 30-question test set (23 English, 7 Bahasa Malaysia; 3 out-of-scope; 8 that need forecast data) in three layers.

Retrieval, with no LLM involved: each question lists the knowledge-base entries that hold its answer.
- **hit rate**: was one of them among the entries retrieved?
- **MRR**: how near the top was it?

Tool choice, also with no LLM: questions are marked with the tools that would answer them, or with "none" for general questions.
- **tool choice accuracy**: did the chatbot call an acceptable tool, and leave the tools alone when the question needed no data?

Answer quality, with one judge call per question returning all metrics as JSON:
- **faithfulness**: no facts beyond the retrieved context
- **answer relevancy**
- **context precision**
- **correctness** against the expected answer
- for the 8 data questions, whose right answer changes every month, the judge is shown what the tools returned and scores **faithfulness** and **relevancy** only: do the answer's numbers match the data?
- **refusal**: for out-of-scope questions, such as an exact future inflation figure or a fund recommendation, the bot must decline rather than invent

A judge reply that cannot be parsed is retried once and then reported as unscored, never counted as zero; if too few questions are scored (a Gemini outage), the run is reported as incomplete, not as a pass or a fail.

It runs in CI against the live API on demand (its own workflow, `rag-eval.yml`, started from the Actions tab) and fails if any judge metric averages below 0.7, the retrieval hit rate falls below 0.8, or tool choice accuracy falls below 0.7. Each run publishes its score table on the workflow's Summary page and uploads the full results, with the judge's reason for every score, as an artifact. The judge model is configurable (`JUDGE_MODEL`), so answers can be graded by a different model from the one that wrote them. It is deliberately not run on every merge: a run makes about 60 to 100 Gemini requests on the same key as the live chatbot, and a day of automatic runs once used up the daily quota and took the chatbot down for users. If `knowledge_base.py` changes, the API re-embeds it at startup, so the chatbot never answers from stale text.

## What is logged

**What users type into the chat is not stored.** The first version of this logged every call, including users' questions, with a notice in the app. I took that out: a household budgeting app has no good reason to keep what people ask it. The API now records an LLM call only when it comes from my own evaluation run.

How it works: the evaluation workflow sends a secret token in an `X-Eval-Token` header. The API compares it with the token it holds (from Secret Manager). On a match, the call writes one row to BigQuery, `observability.llm_calls`. With no token, a wrong token, or no token configured on the server, nothing is written. Tests cover each of those cases.

| What is recorded (evaluation calls only) | Why |
|---|---|
| `retrieval_ms`, `generation_ms`, `total_ms` | Where the time goes: finding context, or the model |
| `prompt_tokens`, `output_tokens`, `total_tokens` | Cost per answer |
| `retrieved_ids`, `retrieved_distances` | Which knowledge-base entries were used, and how close the match was |
| `llm_calls`, `tools_called`, `tool_errors` | How many model calls the answer took, which tools it used, and how many came back as errors |
| `status`, `error_type`, `error_message`, `finish_reason` | Failures and cut-off answers |
| `model`, `api_revision`, `forecast_version` | Which deployment produced the answer |
| `question`, `answer` | The test question and what the chatbot said |

Because every evaluation run asks the same questions, the table shows how latency, token use and retrieval change from one deployment to the next.

Some design decisions:

- **Logging never breaks an answer.** A failed write is caught; the reply still goes out.
- **The row is written before the response is sent.** Cloud Run only gives the container CPU while a request is in flight, so a background write could be lost.
- **Rows expire after 90 days** (day-partitioned table).
- **A test keeps the code and the table in step**: every field the API logs must have a column in the Terraform schema.
- **Gemini failures are a clean 503** with a "try again" message, where they used to be an unhandled 500.

The trade-off: there is no view of real traffic, so a slow or failing day for users shows up only in Cloud Run's own request metrics (status codes and latency, no content). Content-free metrics for real calls would be the next step if that proves too thin.

Example query:

```sql
-- Latency and tokens per deployment, from evaluation runs
SELECT api_revision, DATE(MIN(ts)) AS first_seen,
       COUNT(*) AS calls,
       COUNTIF(status = 'error') AS errors,
       APPROX_QUANTILES(total_ms, 100)[OFFSET(50)] AS p50_ms,
       APPROX_QUANTILES(total_ms, 100)[OFFSET(95)] AS p95_ms,
       ROUND(AVG(total_tokens)) AS avg_tokens
FROM `observability.llm_calls`
GROUP BY api_revision
ORDER BY first_seen DESC;
```
