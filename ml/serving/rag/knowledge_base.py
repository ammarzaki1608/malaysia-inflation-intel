"""
Knowledge base for the RAG layer.

This is the corpus of domain knowledge retrieved to ground Gemini's
answers. It intentionally does NOT contain any live numbers — those
always come from the user's actual /forecast call. This file only
contains stable, factual explanations of concepts, categories, and
methodology.
"""

KNOWLEDGE_BASE = [
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
        "id": "headline_vs_personal_inflation",
        "text": "Malaysia's official headline inflation rate is a national average based on a fixed basket of goods weighted by typical national spending patterns. An individual's personal inflation rate can differ substantially from the headline figure if their actual spending pattern is concentrated more heavily in categories that are rising faster or slower than that national average.",
    },
    {
        "id": "b40_m40_t20",
        "text": "B40, M40, and T20 are Malaysian household income classifications: B40 refers to the bottom 40% of households by income, M40 the middle 40%, and T20 the top 20%. These bands are commonly used in Malaysian policy discussions to describe which income group is affected by a given economic change.",
    },
    {
        "id": "hies_survey",
        "text": "The Household Income and Expenditure Survey (HIES) is a periodic DOSM survey capturing detailed household income and spending patterns. This project uses the 2022 edition — the most recent available at the time of building this system. HIES data is a periodic snapshot, not a continuously updated monthly series.",
    },
    {
        "id": "data_sources",
        "text": "This project's data comes from three official Malaysian sources: the Department of Statistics Malaysia (DOSM) for CPI and Producer Price Index data, Bank Negara Malaysia (BNM) for the Overnight Policy Rate, and DOSM's Household Income and Expenditure Survey for income and expenditure context.",
    },
    {
        "id": "state_level_limitation",
        "text": "CPI and price data in this system are reported at the state level, not the district or city level. This means the figure shown for a large state like Selangor is an average across very different areas — a wealthy suburb and a rural district within the same state will genuinely experience different costs of living, but this system cannot currently distinguish between them.",
    },
]