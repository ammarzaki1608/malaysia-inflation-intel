"""
Versioned evaluation test set for the RAG /ask endpoint.

Each entry defines a question, the expected ground-truth answer
(used by RAGAS to score faithfulness/relevance), and whether the
question requires live forecast data.

IMPORTANT: When you change knowledge_base.py or the /ask prompt,
re-run the evaluation harness against this same test set to check
for regressions before considering the change "done."
"""

TEST_QUESTIONS = [
    {
        "question": "How does the relocation planner suggest destination spending?",
        "ground_truth": "It scales the user's current spending by the current price ratio between the origin and destination states, based on real DOSM CPI index values.",
        "needs_live_data": False,
    },
    {
        "question": "Why can the suggested destination amounts be edited?",
        "ground_truth": "Because the auto-scaling assumes identical spending habits in the new location, which may not hold true if the user's actual habits change, such as renting a smaller home or cooking more at home.",
        "needs_live_data": False,
    },
    {
        "question": "How is the recommended minimum salary calculated?",
        "ground_truth": "It is the projected total destination spending multiplied by 1.30, adding a 30% buffer for savings and unexpected costs.",
        "needs_live_data": False,
    },
    {
        "question": "Why does the forecast become less reliable further into the future?",
        "ground_truth": "Because forecasts are generated recursively — each month's prediction is fed back in as input for the next month, so later predictions increasingly rely on the model's own prior predictions rather than confirmed real data.",
        "needs_live_data": False,
    },
    {
        "question": "What is the single most important feature the forecasting model uses?",
        "ground_truth": "The 3-month moving average of the CPI index (cpi_ma_3), because it smooths out monthly noise and captures the underlying trend more reliably than a single month's raw value.",
        "needs_live_data": False,
    },
    {
        "question": "Why might my personal inflation rate be different from the official headline rate?",
        "ground_truth": "The headline rate is a national average based on a fixed basket of goods, while an individual's actual spending pattern may be concentrated in categories rising faster or slower than that average.",
        "needs_live_data": False,
    },
    {
        "question": "How often is the underlying data refreshed?",
        "ground_truth": "Automatically every month via a scheduled Cloud Run Job that re-downloads the latest DOSM and BNM data and rebuilds all derived tables.",
        "needs_live_data": False,
    },
    {
        "question": "What does B40 mean?",
        "ground_truth": "B40 refers to the bottom 40% of Malaysian households by income.",
        "needs_live_data": False,
    },
    {
        "question": "Why is my biggest leak category rising faster than my other spending?",
        "ground_truth": "The biggest leak category is whichever category in the user's own spending is projected to rise fastest over the chosen horizon, based on the model's forecast for that specific category and state.",
        "needs_live_data": True,
        "state": "Selangor",
        "division": "07",
        "horizon_months": 6,
    },
    {
        "question": "What data source is used for household income context?",
        "ground_truth": "The DOSM Household Income and Expenditure Survey (HIES) 2022 edition, which is a periodic snapshot, not a continuously updated monthly series.",
        "needs_live_data": False,
    },
    {
    "question": "Why is my biggest leak category rising faster than my other spending?",
    "ground_truth": "The biggest leak category is whichever category in the user's own spending is projected to rise fastest over the chosen horizon, based on the model's forecast for that specific category and state.",
    "needs_live_data": True,
    "state": "Selangor",
    "division": "07",
    "horizon_months": 6,
    # NOTE: This question scored 0.50 on answer_relevancy in the first
    # evaluation run (2026-07). The system correctly explains the
    # DEFINITION of "biggest leak" but doesn't provide category-specific
    # economic reasoning (e.g. why Transport specifically is moving).
    # This is a documented, accepted limitation — see knowledge_base.py's
    # "category_specific_causes_limitation" entry. Not treated as a
    # regression as long as it stays a definitional, not causal, answer.
    },
]