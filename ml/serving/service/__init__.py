"""
The pieces behind the API in ml/serving/api.py.

    config       settings, the lists of states and categories, the rate limiter
    data         every BigQuery query
    forecasts    the precomputed forecast table and the published scorecard
    chat         /ask and /prescriptive-insight
    chat_tools   the tools the chatbot may call
    llm_log      logging of LLM calls (evaluation traffic only)

State that changes while the API runs (the forecast table, the BigQuery client,
the Gemini client) lives in exactly one module. Other modules read it through
that module (`forecasts.forecast_rows`), never by importing the name, so a
refreshed table is seen everywhere.
"""
