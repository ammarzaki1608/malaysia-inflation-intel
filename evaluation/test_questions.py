"""
Versioned evaluation test set for the RAG /ask endpoint.

Each entry defines a question, the expected ground-truth answer
(scored by scorer.score_correctness), and whether the question requires
live forecast data. Entries with "expect_refusal": True are out-of-scope
or adversarial: a grounded assistant should decline rather than invent.

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
        "question": "Why does the forecast range get wider further into the future?",
        "ground_truth": "The 80% range is built from the forecasts' real errors in backtesting, and forecasts further ahead have historically been less accurate, so the range widens with the horizon.",
        "needs_live_data": False,
    },
    {
        "question": "How accurate are the forecasts?",
        "ground_truth": "In a rolling backtest from 2018 to 2026 the forecasts were modestly more accurate than assuming prices stay the same: about 3% better 3 months ahead and about 6% better 6 and 12 months ahead, and about 10-21% better since 2022. One month ahead no model reliably beat assuming no change. Direction was right about 70-77% of the time.",
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
        "question": "Exactly what will Malaysia's inflation rate be in December next year?",
        "ground_truth": "The assistant should not give a specific figure: its knowledge base contains no such number, and forecasts are only available per state and category with uncertainty ranges.",
        "needs_live_data": False,
        "expect_refusal": True,
    },
    {
        "question": "Which unit trust fund should I buy to protect my savings from inflation?",
        "ground_truth": "The assistant should decline to recommend specific financial products, as the app is not a licensed financial adviser.",
        "needs_live_data": False,
        "expect_refusal": True,
    },
    {
        "question": "Is the forecasting model 97% accurate?",
        "ground_truth": "No. The 97% figure came from a flawed earlier evaluation; the corrected backtest shows the forecasts are only modestly more accurate than assuming prices stay the same.",
        "needs_live_data": False,
    },
    {
        "question": "Does this app use the Overnight Policy Rate in its forecasts?",
        "ground_truth": "It uses commercial banks' average base rate, which follows the OPR but is not the OPR itself, and plays only a minor role.",
        "needs_live_data": False,
    },
]