"""
Versioned evaluation test set for the RAG /ask endpoint.

Each entry has:
  question          what the user asks (English or Bahasa Malaysia: "lang")
  ground_truth      the expected answer, scored by the judge for correctness
  expected_context  knowledge-base entry ids that hold the answer. Retrieval is
                    scored on these without any LLM: a "hit" means at least one
                    of them is in the top results /ask retrieved. Omitted for
                    out-of-scope questions, where nothing should match.
  needs_live_data   whether the question needs the user's live forecast
                    (then state / division / horizon_months are sent too)
  expect_refusal    True for out-of-scope or adversarial questions: a grounded
                    assistant should decline rather than invent.

IMPORTANT: When you change knowledge_base.py or the /ask prompt,
re-run the evaluation harness against this same test set to check
for regressions before considering the change "done."
tests/evaluation/ checks that every expected_context id still exists.
"""

TEST_QUESTIONS = [
    {
        "question": "How does the relocation planner suggest destination spending?",
        "ground_truth": "It starts from the user's current spending scaled by the difference in average monthly household spending between the two states (DOSM HIES 2022), as a starting point the user can edit. It does not compare the states' CPI index levels.",
        "expected_context": ["relocation_starting_point", "app_relocation_planner_purpose"],
        "needs_live_data": False,
    },
    {
        "question": "Why can the suggested destination amounts be edited?",
        "ground_truth": "Because the starting point reflects the average household in each state, whose lifestyle and income differ from the user's, so the user should adjust each amount to what they actually expect to spend, for example renting a smaller home or cooking at home more.",
        "expected_context": ["relocation_editable_suggestion"],
        "needs_live_data": False,
    },
    {
        "question": "How is the recommended minimum salary calculated?",
        "ground_truth": "It is the projected total destination spending multiplied by 1.30, adding a 30% buffer for savings and unexpected costs.",
        "expected_context": ["relocation_salary_needed"],
        "needs_live_data": False,
    },
    {
        "question": "Why does the forecast range get wider further into the future?",
        "ground_truth": "The 80% range is built from the forecasts' real errors in backtesting, and forecasts further ahead have historically been less accurate, so the range widens with the horizon.",
        "expected_context": ["forecast_ranges", "forecast_horizon_limitation"],
        "needs_live_data": False,
    },
    {
        "question": "How accurate are the forecasts?",
        "ground_truth": "In a rolling backtest from 2018 to 2026 the forecasts were modestly more accurate than assuming prices stay the same: about 3% better 3 months ahead and about 6% better 6 and 12 months ahead, and about 10-21% better since 2022. One month ahead no model reliably beat assuming no change. Direction was right about 70-77% of the time.",
        "expected_context": ["model_accuracy"],
        "needs_live_data": False,
    },
    {
        "question": "Why might my personal inflation rate be different from the official headline rate?",
        "ground_truth": "The headline rate is a national average based on a fixed basket of goods, while an individual's actual spending pattern may be concentrated in categories rising faster or slower than that average.",
        "expected_context": ["headline_vs_personal_inflation"],
        "needs_live_data": False,
    },
    {
        "question": "How often is the underlying data refreshed?",
        "ground_truth": "Automatically every month via a scheduled Cloud Run Job that re-downloads the latest DOSM and BNM data and rebuilds all derived tables.",
        "expected_context": ["automated_pipeline"],
        "needs_live_data": False,
    },
    {
        "question": "What does B40 mean?",
        "ground_truth": "B40 refers to the bottom 40% of Malaysian households by income.",
        "expected_context": ["b40_m40_t20"],
        "needs_live_data": False,
    },
    {
        "question": "Why is my biggest leak category rising faster than my other spending?",
        "ground_truth": "The biggest leak category is whichever category in the user's own spending is projected to rise fastest over the chosen horizon, based on the model's forecast for that specific category and state.",
        "expected_context": ["biggest_leak_explainer", "category_specific_causes_limitation"],
        "needs_live_data": True,
        "state": "Selangor",
        "division": "07",
        "horizon_months": 6,
    },
    {
        "question": "What data source is used for household income context?",
        "ground_truth": "The DOSM Household Income and Expenditure Survey (HIES) 2022 edition, which is a periodic snapshot, not a continuously updated monthly series.",
        "expected_context": ["hies_survey", "data_sources"],
        "needs_live_data": False,
    },
    {
        "question": "Is the forecasting model 97% accurate?",
        "ground_truth": "No. The 97% figure came from a flawed earlier evaluation; the corrected backtest shows the forecasts are only modestly more accurate than assuming prices stay the same.",
        "expected_context": ["previous_model_correction"],
        "needs_live_data": False,
    },
    {
        "question": "Does this app use the Overnight Policy Rate in its forecasts?",
        "ground_truth": "It uses commercial banks' average base rate, which follows the OPR but is not the OPR itself, and plays only a minor role.",
        "expected_context": ["feature_interest_rate"],
        "needs_live_data": False,
    },
    {
        "question": "What does the Transport category cover?",
        "ground_truth": "Transport (CPI Division 07) covers fuel, public transport fares, and vehicle purchase and maintenance costs, and is directly affected by fuel subsidy policy changes.",
        "expected_context": ["cpi_division_07"],
        "needs_live_data": False,
    },
    {
        "question": "Is the price data available for my district or city?",
        "ground_truth": "No. CPI and price data are reported at the state level only, so a state's figure is an average across very different areas.",
        "expected_context": ["state_level_limitation"],
        "needs_live_data": False,
    },
    {
        "question": "What information does the forecasting model use to make a prediction?",
        "ground_truth": "Only information available at the forecast date: recent price changes in that state and category over several periods, the long-run average change, how often prices stayed unchanged, how the category is moving in other states, last year's same-period change, producer price changes and the interest rate.",
        "expected_context": ["forecast_features", "how_forecasts_are_made"],
        "needs_live_data": False,
    },

    # ── Bahasa Malaysia: the app is bilingual, so the chatbot is tested in both ──
    {
        "question": "Apakah maksud B40?",
        "lang": "ms",
        "ground_truth": "B40 refers to the bottom 40% of Malaysian households by income.",
        "expected_context": ["b40_m40_t20"],
        "needs_live_data": False,
    },
    {
        "question": "Berapa kerapkah data dikemas kini?",
        "lang": "ms",
        "ground_truth": "Automatically every month via a scheduled Cloud Run Job that re-downloads the latest DOSM and BNM data and rebuilds all derived tables.",
        "expected_context": ["automated_pipeline"],
        "needs_live_data": False,
    },
    {
        "question": "Bagaimanakah gaji minimum yang disyorkan dikira?",
        "lang": "ms",
        "ground_truth": "It is the projected total destination spending multiplied by 1.30, adding a 30% buffer for savings and unexpected costs.",
        "expected_context": ["relocation_salary_needed"],
        "needs_live_data": False,
    },
    {
        "question": "Mengapakah julat ramalan semakin lebar untuk tempoh yang lebih jauh?",
        "lang": "ms",
        "ground_truth": "The 80% range is built from the forecasts' real errors in backtesting, and forecasts further ahead have historically been less accurate, so the range widens with the horizon.",
        "expected_context": ["forecast_ranges", "forecast_horizon_limitation"],
        "needs_live_data": False,
    },

    # ── Out of scope: the assistant must decline, not invent ──
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
        "question": "Dana unit amanah manakah yang patut saya beli untuk melindungi simpanan saya daripada inflasi?",
        "lang": "ms",
        "ground_truth": "The assistant should decline to recommend specific financial products, as the app is not a licensed financial adviser.",
        "needs_live_data": False,
        "expect_refusal": True,
    },
]
