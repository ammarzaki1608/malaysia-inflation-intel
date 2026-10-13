"""
Knowledge base for the RAG layer.

This is the corpus of domain knowledge retrieved to ground Gemini's
answers. It intentionally does NOT contain any live numbers — those
always come from the user's actual /forecast call. This file only
contains stable, factual explanations of concepts, categories, and
methodology.
"""

KNOWLEDGE_BASE = [
    # ── CPI categories ───────────────────────────────────────────────────────
    {
        "id": "cpi_division_01",
        "text": "Food & Non-Alcoholic Beverages (CPI Division 01) covers groceries, fresh produce, meat, fish, dairy, and non-alcoholic drinks purchased for home consumption. It does not include restaurant meals, which fall under Division 11.",
    },
    {
        "id": "cpi_division_04",
        "text": "Housing, Water, Electricity & Gas (CPI Division 04) covers rent, utility bills, and home maintenance costs. It is typically the largest or second-largest category in most Malaysian households' budgets.",
    },
    {
        "id": "cpi_division_07",
        "text": "Transport (CPI Division 07) covers fuel, public transport fares, vehicle purchase and maintenance costs. This category is directly affected by fuel subsidy policy changes.",
    },
    {
        "id": "cpi_division_11",
        "text": "Restaurants & Accommodation (CPI Division 11) covers eating out, hotel stays, and related services — distinct from Division 01, which only covers groceries for home cooking.",
    },

    # ── Forecasting model & features ────────────────────────────────────────
    {
        "id": "how_forecasts_are_made",
        "text": "Forecasts come from an average of two models: a per-series exponential smoothing model (ETS) and an XGBoost model that predicts the percentage change in each state's category price index directly for each horizon from 1 to 12 months ahead. Predictions are never fed back in as inputs. Both models are retrained every month on all available data, and the forecasts for every state and category are computed once and stored, so every user sees the same forecast for the same state and category.",
    },
    {
        "id": "forecast_features",
        "text": "The XGBoost model uses only information available at the forecast date: recent price changes in that state and category over 1, 3, 6, 12 and 24 months; the long-run average monthly change; how often prices in that series stayed unchanged over the past year (many categories change in steps rather than every month); how the same category is moving across all states; last year's change over the same calendar months; changes in the Producer Price Index; and the commercial banks' base lending rate. It also knows which state and category it is forecasting, so flat categories like Clothing are not given the same upward drift as faster-rising ones like Food.",
    },
    {
        "id": "feature_ppi_lag",
        "text": "Producer Price Index (PPI) changes are used because producer prices can lead consumer prices: a rise in what factories and wholesalers pay may show up in consumer prices one to three months later. The model uses PPI changes over the previous 1, 3 and 12 months.",
    },
    {
        "id": "feature_interest_rate",
        "text": "The interest-rate input is the average base rate of Malaysian commercial banks, published by Bank Negara Malaysia (BNM). The base rate follows BNM's Overnight Policy Rate (OPR) but is not the OPR itself, and the series is only available from about 2022, so it plays a minor role in the forecasts.",
    },
    {
        "id": "model_accuracy",
        "text": "The forecasts were tested with a rolling backtest: for every month from January 2018 to July 2026, the models were retrained using only data available at that date and their forecasts were compared with what actually happened. Compared with simply assuming prices stay the same, the forecast was about 3% more accurate 3 months ahead and about 6% more accurate 6 and 12 months ahead across the whole period, and about 10-21% more accurate since 2022. One month ahead, no model reliably beat assuming no change, because about 40% of monthly category price changes are exactly zero. The direction of price change was predicted correctly about 70-77% of the time.",
    },
    {
        "id": "forecast_ranges",
        "text": "Each forecast comes with an 80% range: the true value is expected to fall inside it about 8 times out of 10. The range is built from the forecasts' real errors over the most recent 36 months of backtesting, so it widens with the horizon because longer forecasts have been less accurate. When tested on data the range was not built from, it contained the actual outcome 81% of the time. All categories currently share the same range width for a given horizon.",
    },
    {
        "id": "previous_model_correction",
        "text": "An earlier version of this app reported 97% directional accuracy and a mean absolute error of 0.73. Those figures were wrong: the evaluation accidentally let the model see the month it was predicting and only tested one month ahead. A corrected evaluation showed that the earlier model was less accurate than simply assuming prices stay the same at every horizon (about twice as inaccurate one month ahead), so it was replaced with the current models.",
    },

    # ── Personal vs headline inflation ──────────────────────────────────────
    {
        "id": "headline_vs_personal_inflation",
        "text": "Malaysia's official headline inflation rate is a national average based on a fixed basket of goods weighted by typical national spending patterns. An individual's personal inflation rate can differ substantially from the headline figure if their actual spending pattern is concentrated more heavily in categories that are rising faster or slower than that national average. The app compares like with like: your personal inflation over the past 12 months (each category's real price change in your state, weighted by how much you spend on it) against the official rate over the same 12 months. The forecast for the coming months is shown separately as an outlook.",
    },
    {
        "id": "biggest_leak_explainer",
        "text": "The 'biggest leak' category shown in a personal forecast report is whichever category in the user's own spending is projected to rise the fastest over their chosen horizon. It is calculated by comparing the projected percentage change of each category the user has entered spending for, and highlighting the one rising fastest — not necessarily the category with the largest RM amount.",
    },
    {
        "id": "b40_m40_t20",
        "text": "B40, M40, and T20 are Malaysian household income classifications: B40 refers to the bottom 40% of households by income, M40 the middle 40%, and T20 the top 20%. These bands are commonly used in Malaysian policy discussions to describe which income group is affected by a given economic change.",
    },

    # ── Relocation Planner logic ────────────────────────────────────────────
    {
        "id": "relocation_starting_point",
        "text": "The Relocation Planner's suggested destination amounts start from the user's current spending, scaled by the difference in average monthly household spending between the two states from DOSM's Household Income and Expenditure Survey 2022. That difference reflects lifestyle and income as well as prices, so it is only a starting point. The app does not compare states' CPI index levels, because each state's index measures how prices changed since its own base period, not how expensive the state is.",
    },
    {
        "id": "relocation_editable_suggestion",
        "text": "Suggested destination spending amounts in the Relocation Planner are editable because the auto-scaling assumes the user keeps identical spending habits in the new location, which may not hold true — for example, someone might rent a smaller home, or cook at home more often, changing their actual spending pattern beyond what an average-household comparison would predict.",
    },
    {
        "id": "relocation_salary_needed",
        "text": "The 'recommended minimum salary' shown in the Relocation Planner is calculated as the projected total destination spending multiplied by 1.30, adding a 30% buffer intended to account for savings and unexpected costs when moving to a new state.",
    },

    # ── Data sources & pipeline ──────────────────────────────────────────────
    {
        "id": "data_sources",
        "text": "This project's data comes from three official Malaysian sources: the Department of Statistics Malaysia (DOSM) for CPI and Producer Price Index data, Bank Negara Malaysia (BNM) for commercial banks' base lending rates, and DOSM's Household Income and Expenditure Survey for income and expenditure context.",
    },
    {
        "id": "hies_survey",
        "text": "The Household Income and Expenditure Survey (HIES) is a periodic DOSM survey capturing detailed household income and spending patterns. This project uses the 2022 edition — the most recent available at the time of building this system. HIES data is a periodic snapshot, not a continuously updated monthly series.",
    },
    {
        "id": "automated_pipeline",
        "text": "This system's underlying data is refreshed automatically every month via a scheduled Cloud Run Job, which re-downloads the latest official DOSM and BNM data, reloads it into the database, rebuilds all derived data tables, retrains the forecasting models and publishes new forecasts. This means forecasts and historical charts reflect the most recently published government data without requiring any manual update.",
    },

    # ── App usage & features ────────────────────────────────────────────────
    {
        "id": "app_bilingual",
        "text": "This application is available in both English and Bahasa Malaysia. The language can be switched at any time using the toggle at the top of the sidebar, and all pages, including forecasts and reports, update immediately to the selected language.",
    },
    {
        "id": "app_personal_calculator_purpose",
        "text": "The Personal Inflation Calculator lets a user enter their monthly salary and actual spending per category in ringgit, then shows their personal inflation over the past 12 months compared with the official rate, and a forecast of how that spending is projected to change over 3, 6, or 12 months with an 80% range.",
    },
    {
        "id": "app_relocation_planner_purpose",
        "text": "The Relocation Planner helps a user evaluate moving from one Malaysian state to another by starting from the difference in average household spending between the two states, letting the user adjust each amount, forecasting how prices will change in each state, and estimating the salary needed in the new state to maintain the same standard of living.",
    },
    {
        "id": "app_data_explorer_purpose",
        "text": "The Data Explorer page lets users browse the real historical CPI, Producer Price Index, and interest rate data behind the app's forecasts, by state, category, and time range, without needing to enter any personal financial information.",
    },

    # ── Known limitations ────────────────────────────────────────────────────
    {
        "id": "state_level_limitation",
        "text": "CPI and price data in this system are reported at the state level, not the district or city level. This means the figure shown for a large state like Selangor is an average across very different areas — a wealthy suburb and a rural district within the same state will genuinely experience different costs of living, but this system cannot currently distinguish between them.",
    },
    {
        "id": "forecast_horizon_limitation",
        "text": "Consumer prices in Malaysia change slowly and often stay flat from one month to the next, so short-term inflation is only modestly predictable. Forecasts further ahead are more uncertain, which is why the 80% range widens with the horizon. Users should treat forecasts as an indication of direction and rough size rather than precise figures, and pay attention to the range.",
    },
    {
    "id": "category_specific_causes_limitation",
    "text": "This system can explain how forecast figures and labels like 'biggest leak' are calculated, and can point to the model's general inputs (such as recent price changes, how the category is moving in other states, or producer price trends). It does not currently have category-specific economic narratives explaining why one particular category (e.g. Transport or Insurance) is moving the way it is beyond these general model mechanics.",
    },
]