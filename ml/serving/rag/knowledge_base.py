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
        "id": "feature_cpi_ma_3",
        "text": "cpi_ma_3 is the 3-month moving average of the CPI index. It is the single most important feature in the forecasting model, because it smooths out one-off monthly noise and captures the underlying trend direction more reliably than any single month's raw value.",
    },
    {
        "id": "feature_ppi_lag",
        "text": "Producer Price Index (PPI) features are included because producer prices tend to lead consumer prices — a rise in what factories and wholesalers pay often shows up in consumer prices one to three months later. This is why the model uses ppi_lag_1, ppi_lag_2, and ppi_lag_3 as leading indicators.",
    },
    {
        "id": "feature_opr",
        "text": "The Overnight Policy Rate (OPR) is Bank Negara Malaysia's benchmark interest rate. It influences borrowing costs, consumer spending, and indirectly, inflation. The model uses it as a monetary-policy signal alongside price-based features.",
    },
    {
        "id": "recursive_forecast_explainer",
        "text": "Multi-month forecasts are generated recursively: the model predicts one month ahead, then that prediction is fed back in as input to predict the next month, repeated for the requested horizon. Because later predictions partly rely on earlier predictions rather than confirmed real data, accuracy naturally decreases the further into the future the forecast extends — this is why the confidence range widens with each additional month.",
    },
    {
        "id": "model_accuracy",
        "text": "The forecasting model is an XGBoost model achieving 97% directional accuracy and a mean absolute error of 0.73 on historical validation, evaluated using walk-forward cross-validation — a method appropriate for time-series data that always trains on the past and tests on the future, avoiding the data leakage that a standard random train/test split would introduce.",
    },

    # ── Personal vs headline inflation ──────────────────────────────────────
    {
        "id": "headline_vs_personal_inflation",
        "text": "Malaysia's official headline inflation rate is a national average based on a fixed basket of goods weighted by typical national spending patterns. An individual's personal inflation rate can differ substantially from the headline figure if their actual spending pattern is concentrated more heavily in categories that are rising faster or slower than that national average.",
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
        "id": "relocation_price_ratio",
        "text": "The Relocation Planner suggests destination spending amounts by scaling the user's current spending using the current price ratio between the two states for each category — calculated from real DOSM CPI index values, not an estimate. For example, if a category's index is 11.5% higher in the destination state, the suggested amount for that category is scaled up by 11.5%.",
    },
    {
        "id": "relocation_editable_suggestion",
        "text": "Suggested destination spending amounts in the Relocation Planner are editable because the auto-scaling assumes the user keeps identical spending habits in the new location, which may not hold true — for example, someone might rent a smaller home, or cook at home more often, changing their actual spending pattern beyond what a simple price-ratio scaling would predict.",
    },
    {
        "id": "relocation_salary_needed",
        "text": "The 'recommended minimum salary' shown in the Relocation Planner is calculated as the projected total destination spending multiplied by 1.30, adding a 30% buffer intended to account for savings and unexpected costs when moving to a new state.",
    },

    # ── Data sources & pipeline ──────────────────────────────────────────────
    {
        "id": "data_sources",
        "text": "This project's data comes from three official Malaysian sources: the Department of Statistics Malaysia (DOSM) for CPI and Producer Price Index data, Bank Negara Malaysia (BNM) for the Overnight Policy Rate, and DOSM's Household Income and Expenditure Survey for income and expenditure context.",
    },
    {
        "id": "hies_survey",
        "text": "The Household Income and Expenditure Survey (HIES) is a periodic DOSM survey capturing detailed household income and spending patterns. This project uses the 2022 edition — the most recent available at the time of building this system. HIES data is a periodic snapshot, not a continuously updated monthly series.",
    },
    {
        "id": "automated_pipeline",
        "text": "This system's underlying data is refreshed automatically every month via a scheduled Cloud Run Job, which re-downloads the latest official DOSM and BNM data, reloads it into the database, and rebuilds all derived data tables. This means forecasts and historical charts reflect the most recently published government data without requiring any manual update.",
    },

    # ── App usage & features ────────────────────────────────────────────────
    {
        "id": "app_bilingual",
        "text": "This application is available in both English and Bahasa Malaysia. The language can be switched at any time using the toggle at the top of the sidebar, and all pages, including forecasts and reports, update immediately to the selected language.",
    },
    {
        "id": "app_personal_calculator_purpose",
        "text": "The Personal Inflation Calculator lets a user enter their monthly salary and actual spending per category in ringgit, then generates a forecast of how that spending is projected to change over 3, 6, or 12 months, compared against the official national inflation rate.",
    },
    {
        "id": "app_relocation_planner_purpose",
        "text": "The Relocation Planner helps a user evaluate moving from one Malaysian state to another by comparing their cost of living between the two states, using real price differences, and estimating the salary needed in the new state to maintain the same standard of living.",
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
        "id": "recursive_forecast_limitation",
        "text": "Because multi-month forecasts are generated recursively, longer-horizon forecasts (such as 12 months) are less reliable than shorter ones (such as 3 months), since later predictions are increasingly built on the model's own earlier predictions rather than confirmed real data. Users should treat longer-horizon forecasts as directional guidance rather than precise figures.",
    },
]