"""
Malaysia Personal Inflation Intelligence — Streamlit App

A citizen-facing tool that calculates personal inflation rates
and helps Malaysians plan relocation decisions using official
DOSM and BNM data.
"""

import httpx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

# ── page config — must be first Streamlit call ───────────────────────────────
st.set_page_config(
    page_title="Malaysia Inflation Intelligence",
    page_icon="🇲🇾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8080")

CATEGORIES = {
    "01": "Food & Non-Alcoholic Beverages",
    "02": "Alcoholic Beverages & Tobacco",
    "03": "Clothing & Footwear",
    "04": "Housing, Water, Electricity & Gas",
    "05": "Furnishings & Household Equipment",
    "06": "Health",
    "07": "Transport",
    "08": "Information & Communication",
    "09": "Recreation, Sport & Culture",
    "10": "Education",
    "11": "Restaurants & Accommodation",
    "12": "Insurance & Financial Services",
    "13": "Personal Care & Miscellaneous",
}

STATES = [
    "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
    "Pahang", "Perak", "Perlis", "Pulau Pinang", "Sabah",
    "Sarawak", "Selangor", "Terengganu",
    "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya"
]

# ── default spending profiles ─────────────────────────────────────────────────
DEFAULT_PROFILES = {
    "Fresh Graduate (Urban)": {
        "01": 30, "04": 25, "07": 10, "11": 15,
        "10": 5,  "06": 5,  "03": 5,  "09": 5,
    },
    "Young Family (Suburban)": {
        "01": 30, "04": 25, "07": 15, "11": 5,
        "10": 10, "06": 5,  "03": 5,  "09": 5,
    },
    "Single Professional (City)": {
        "01": 20, "04": 30, "07": 10, "11": 20,
        "10": 5,  "06": 5,  "03": 5,  "09": 5,
    },
    "Rural Household": {
        "01": 40, "04": 20, "07": 20, "11": 5,
        "10": 5,  "06": 5,  "03": 3,  "09": 2,
    },
    "Custom": {},
}

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F4E79;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 4px;
    }
    .positive { color: #dc2626; }
    .negative { color: #16a34a; }
    .summary-box {
        background: #EFF6FF;
        border-left: 4px solid #1F4E79;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── helper functions ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_price_ratio(origin_state: str, destination_state: str, division: str) -> dict:
    """Get the current price ratio between two states for one category."""
    try:
        response = httpx.get(
            f"{API_URL}/price-ratio",
            params={
                "origin_state": origin_state,
                "destination_state": destination_state,
                "division": division,
            },
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def fetch_prediction(state: str, weights: dict) -> dict:
    """
    Call the FastAPI /predict endpoint.
    Cached for 5 minutes to avoid hammering the API on every rerun.
    """
    try:
        response = httpx.post(
            f"{API_URL}/predict",
            json={"state": state, "weights": weights},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        st.error("Request timed out. The API may be loading the model.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_comparison(
    origin_state: str,
    destination_state: str,
    origin_weights: dict,
    destination_weights: dict,
) -> dict:
    """Call the FastAPI /compare endpoint."""
    try:
        response = httpx.post(
            f"{API_URL}/compare",
            json={
                "origin_state":        origin_state,
                "destination_state":   destination_state,
                "origin_weights":      origin_weights,
                "destination_weights": destination_weights,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        st.error("Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_forecast(state: str, division: str, horizon_months: int) -> dict:
    """
    Call the FastAPI /forecast endpoint for a multi-month
    recursive prediction of one category.
    """
    try:
        response = httpx.get(
            f"{API_URL}/forecast",
            params={
                "state": state,
                "division": division,
                "horizon_months": horizon_months,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        st.error("Forecast request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def weights_to_str_keys(weights: dict) -> dict:
    """Convert integer keys to string keys for API compatibility."""
    return {str(k): float(v) for k, v in weights.items() if v > 0}


def render_inflation_gauge(value: float, title: str) -> go.Figure:
    """
    Render a gauge chart showing personal inflation rate.
    Green = deflation, Red = high inflation.
    """
    color = "#dc2626" if value > 0 else "#16a34a"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": title, "font": {"size": 14, "color": "#1F4E79"}},
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [-3, 5], "ticksuffix": "%"},
            "bar":  {"color": color},
            "steps": [
                {"range": [-3, 0],  "color": "#dcfce7"},
                {"range": [0, 2],   "color": "#fef9c3"},
                {"range": [2, 3.5], "color": "#ffedd5"},
                {"range": [3.5, 5], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "#1F4E79", "width": 2},
                "thickness": 0.75,
                "value": 0,
            },
        }
    ))

    fig.update_layout(
        height=220,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_category_waterfall(categories: list) -> go.Figure:
    """
    Waterfall chart showing each category's contribution
    to personal inflation rate.
    """
    sorted_cats = sorted(
        categories,
        key=lambda x: x["weighted_contribution"],
        reverse=True,
    )

    names  = [c["category_name"].replace(" & ", " &\n") for c in sorted_cats]
    values = [c["weighted_contribution"] for c in sorted_cats]
    colors = ["#dc2626" if v > 0 else "#16a34a" for v in values]

    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        marker_color=colors,
        text=[f"{v:+.3f}%" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Contribution to your personal inflation rate by category",
        yaxis_title="Weighted contribution (%)",
        xaxis_tickangle=-30,
        height=380,
        margin=dict(t=50, b=120, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig

def render_forecast_chart(forecast_data: list, category_name: str, base_amount_rm: float = None) -> go.Figure:
    """
    Render a multi-month forecast with a widening confidence band.

    If base_amount_rm is provided, converts the CPI index forecast
    into an RM trend line by scaling proportionally to the index change.
    Otherwise shows the raw CPI index values.
    """
    steps = [f"Month {f['step']}" for f in forecast_data]

    if base_amount_rm is not None:
        # Convert index values to RM by scaling relative to the
        # first prediction's index — this turns an abstract index
        # into "how much you'll actually spend"
        first_index = forecast_data[0]["predicted_index"]
        values      = [base_amount_rm * (f["predicted_index"] / first_index) for f in forecast_data]
        lower       = [base_amount_rm * (f["lower_bound"]      / first_index) for f in forecast_data]
        upper       = [base_amount_rm * (f["upper_bound"]      / first_index) for f in forecast_data]
        y_title = "Projected spending (RM)"
        hover_suffix = ""
    else:
        values = [f["predicted_index"] for f in forecast_data]
        lower  = [f["lower_bound"] for f in forecast_data]
        upper  = [f["upper_bound"] for f in forecast_data]
        y_title = "CPI Index"
        hover_suffix = ""

    fig = go.Figure()

    # Confidence band — drawn first so the line sits on top of it
    fig.add_trace(go.Scatter(
        x=steps + steps[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(31, 78, 121, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="Confidence range",
    ))

    # Main forecast line
    fig.add_trace(go.Scatter(
        x=steps,
        y=values,
        mode="lines+markers",
        line=dict(color="#1F4E79", width=3),
        marker=dict(size=8),
        name=category_name,
    ))

    fig.update_layout(
        title=f"{category_name} — {len(forecast_data)}-month forecast",
        yaxis_title=y_title,
        height=380,
        margin=dict(t=50, b=40, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    return fig

def render_shap_bar(shap_data: list, title: str) -> go.Figure:
    """
    Horizontal bar chart showing SHAP feature importance.
    """
    features = [s["feature"] for s in shap_data]
    values   = [s["importance"] for s in shap_data]

    feature_labels = {
        "cpi_ma_3":   "3-month CPI trend",
        "cpi_lag_1":  "Last month CPI",
        "cpi_lag_3":  "3-month ago CPI",
        "cpi_lag_6":  "6-month ago CPI",
        "cpi_lag_12": "12-month ago CPI",
        "cpi_ma_12":  "12-month CPI trend",
        "ppi_index":  "Producer prices (current)",
        "ppi_lag_1":  "Producer prices (1mo ago)",
        "ppi_lag_2":  "Producer prices (2mo ago)",
        "ppi_lag_3":  "Producer prices (3mo ago)",
        "opr_rate":   "BNM interest rate",
    }

    labels = [feature_labels.get(f, f) for f in features]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color="#1F4E79",
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Importance score",
        height=280,
        margin=dict(t=50, b=20, l=180, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"autorange": "reversed"},
    )
    return fig


def render_spending_profile_pie(weights: dict) -> go.Figure:
    """Pie chart of the user's spending profile."""
    labels = [CATEGORIES.get(k, k) for k in weights if weights[k] > 0]
    values = [weights[k] for k in weights if weights[k] > 0]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo="label+percent",
        textfont_size=11,
    ))

    fig.update_layout(
        height=320,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig

# ── sidebar — spending profile builder ───────────────────────────────────────

def render_sidebar() -> tuple:
    """
    Renders the sidebar with state selector and spending profile sliders.
    Returns (state, weights, mode) where mode is 'personal' or 'relocation'.
    """
    st.sidebar.title("🇲🇾 Malaysia Inflation Intelligence")
    st.sidebar.caption("Built on official DOSM & BNM data")
    st.sidebar.divider()

    # Mode selector
    mode = st.sidebar.radio(
    "Choose mode",
    options=["🏠 Home", "📊 Personal Inflation Calculator", "🗺️ Relocation Planner"],
        help="Personal: calculate your current inflation rate. Relocation: compare two states."
    )
    st.sidebar.divider()

    # State selector
    state = st.sidebar.selectbox(
        "Your state" if mode == "Personal Inflation Calculator" else "Origin state (where you live now)",
        options=STATES,
        index=STATES.index("Kedah"),
    )

    st.sidebar.divider()

    # ── salary + RM spending inputs (replaces % sliders) ──────────────────────
    st.sidebar.subheader("💰 Your monthly budget")
    st.sidebar.caption("Enter real amounts — we'll do the maths.")

    monthly_salary = st.sidebar.number_input(
        "Monthly salary (RM)",
        min_value=0,
        max_value=50000,
        value=4500,
        step=100,
    )

    st.sidebar.markdown("**Spending per category (RM):**")

    spending = {}
    total_spent = 0

    default_spending = {
        "01": 900, "04": 700, "07": 400, "11": 300,
        "10": 150, "06": 150, "03": 150, "09": 150,
    }

    for div, name in CATEGORIES.items():
        default_val = default_spending.get(div, 0)
        val = st.sidebar.number_input(
            name,
            min_value=0,
            max_value=10000,
            value=default_val,
            step=50,
            key=f"spend_{div}",
        )
        spending[div] = val
        total_spent += val

    unallocated = monthly_salary - total_spent

    st.sidebar.divider()
    st.sidebar.metric("Allocated", f"RM{total_spent:,}")
    st.sidebar.metric("Unallocated (savings)", f"RM{unallocated:,}")

    if unallocated < 0:
        st.sidebar.error(
            f"You've allocated RM{abs(unallocated):,} more than your salary."
        )

    st.sidebar.divider()

    horizon_months = st.sidebar.radio(
        "Forecast horizon",
        options=[3, 6, 12],
        index=1,
        format_func=lambda x: f"{x} months",
        horizontal=True,
    )

    return state, spending, monthly_salary, horizon_months, mode

def render_home():
    """Landing page — explains what the app does and how to use it."""

    # ── hero section ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 40px 20px 20px 20px;'>
        <div style='font-size:3rem;'>🇲🇾</div>
        <h1 style='color:#1F4E79; font-size:2.2rem; margin-bottom:8px;'>
            Malaysia Personal Inflation Intelligence
        </h1>
        <p style='color:#64748b; font-size:1.1rem; max-width:620px; margin:0 auto;'>
            The government publishes one inflation number for the whole country.
            But your inflation depends on <em>how you actually spend your money</em>.
            This tool calculates yours.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── what is personal inflation ─────────────────────────────────────────────
    st.subheader("💡 What is personal inflation?")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        Malaysia's official CPI is calculated using a **fixed national basket** —
        an assumed average of how Malaysians spend money across 13 categories.

        But if you spend **40% of your income on food** and food prices rise 5%,
        you feel that much more than someone who spends only 20% on food.

        **Your inflation = your spending pattern × actual price changes per category.**
        """)

    with col2:
        st.markdown("""
        <div style='background:#EFF6FF; border-left:4px solid #1F4E79;
                    border-radius:8px; padding:16px 20px;'>
            <b style='color:#1F4E79'>Example</b><br><br>
            Government headline CPI: <b>2.1%</b><br><br>
            Person A (heavy food spender): <b>2.8%</b> 📈<br>
            Person B (heavy housing spender): <b>1.7%</b> 📉<br><br>
            <span style='color:#64748b; font-size:0.9rem'>
            Same month. Same country. Different reality.
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── what you can do ────────────────────────────────────────────────────────
    st.subheader("🛠️ What you can do with this app")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div style='background:#F8FAFC; border:1px solid #E2E8F0;
                    border-radius:12px; padding:24px;'>
            <div style='font-size:2rem'>📊</div>
            <h3 style='color:#1F4E79; margin:8px 0;'>Budget Forecast</h3>
            <p style='color:#475569;'>
                Enter your salary and how much you spend per category in
                ringgit. We project your spending 3, 6, or 12 months
                ahead, show you how it compares to official inflation,
                and flag which category is quietly eating into your
                budget the most.
            </p>
            <p style='color:#1F4E79; font-weight:600'>
                Best for: seeing what your money actually buys next year.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style='background:#F8FAFC; border:1px solid #E2E8F0;
                    border-radius:12px; padding:24px;'>
            <div style='font-size:2rem'>🗺️</div>
            <h3 style='color:#1F4E79; margin:8px 0;'>Relocation Planner</h3>
            <p style='color:#475569;'>
                Thinking of moving to KL, Johor, or another state?
                We auto-scale your current spending using real price
                differences between states, forecast both side by side
                in ringgit, and estimate the salary you'd need to keep
                your standard of living.
            </p>
            <p style='color:#1F4E79; font-weight:600'>
                Best for: evaluating a job offer in another state.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── how to use ────────────────────────────────────────────────────────────
    st.subheader("📖 How to use this app — 3 steps")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("""
        <div style='text-align:center; padding:16px;'>
            <div style='font-size:2.5rem'>1️⃣</div>
            <h4 style='color:#1F4E79'>Pick your mode</h4>
            <p style='color:#475569; font-size:0.9rem'>
                Choose <b>Personal Inflation Calculator</b> or
                <b>Relocation Planner</b> from the sidebar on the left.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div style='text-align:center; padding:16px;'>
            <div style='font-size:2.5rem'>2️⃣</div>
            <h4 style='color:#1F4E79'>Enter your real numbers</h4>
            <p style='color:#475569; font-size:0.9rem'>
                Type in your monthly salary and how much you actually
                spend per category in <b>ringgit</b> — no percentages,
                no guesswork. Pick how far ahead you want to forecast:
                3, 6, or 12 months.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div style='text-align:center; padding:16px;'>
            <div style='font-size:2.5rem'>3️⃣</div>
            <h4 style='color:#1F4E79'>Read your results</h4>
            <p style='color:#475569; font-size:0.9rem'>
                See how your spending compares to official inflation,
                which category is your biggest leak, and what your
                monthly surplus looks like months from now.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── data and model ────────────────────────────────────────────────────────
    st.subheader("📡 Where does the data come from?")

    st.markdown("""
    All data is sourced from official Malaysian government APIs — no third-party
    or estimated data. Updated monthly.
    """)

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        st.markdown("""
        <div style='background:#LTTEAL; padding:16px; border-radius:8px;
                    border:1px solid #D1FAE5;'>
            <b style='color:#065F46'>📊 DOSM</b><br>
            <span style='color:#475569; font-size:0.9rem'>
            Department of Statistics Malaysia —
            monthly CPI by state and category,
            Producer Price Index, and household
            income survey data.
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown("""
        <div style='background:#EFF6FF; padding:16px; border-radius:8px;
                    border:1px solid #BFDBFE;'>
            <b style='color:#1E40AF'>🏦 Bank Negara Malaysia</b><br>
            <span style='color:#475569; font-size:0.9rem'>
            Monthly interest rate data including
            the Overnight Policy Rate (OPR) —
            used as an economic signal in
            the forecast model.
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_d3:
        st.markdown("""
        <div style='background:#FFF7ED; padding:16px; border-radius:8px;
                    border:1px solid #FED7AA;'>
            <b style='color:#92400E'>🤖 XGBoost ML Model</b><br>
            <span style='color:#475569; font-size:0.9rem'>
            Trained on 2010–2026 data across
            208 state-category series.
            97% accuracy in predicting the
            direction of inflation movement.
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── disclaimer ─────────────────────────────────────────────────────────────
    with st.expander("⚠️ Important disclaimer — please read"):
        st.markdown("""
        - This tool uses **state-level average data** from DOSM. Actual prices
          vary by neighbourhood, shop, and lifestyle — this is a planning guide,
          not a guarantee.
        - Multi-month forecasts use **recursive prediction** — each month's
          forecast is partly built on the previous month's forecast, so
          accuracy naturally decreases the further out you look. Treat
          12-month forecasts as directional, not precise.
        - Destination spending in the Relocation Planner is **auto-scaled
          using today's price ratio** between states — it assumes you'll
          keep the same spending habits, which may not hold if your
          lifestyle changes significantly.
        - Income and expenditure figures come from the **HIES 2022 survey**
          and may not reflect current conditions precisely.
        - The official inflation figure shown for comparison is currently
          a **fixed reference value**, not a live monthly figure — this
          will be replaced with a live national CPI feed in a future update.
        - This is an **independent research project** — not affiliated with
          DOSM, BNM, or any government agency.
        """)

    st.divider()

    # ── CTA ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:20px;'>
        <p style='color:#475569; font-size:1rem; margin-bottom:16px;'>
            Ready to find out your personal inflation rate?
        </p>
        <p style='color:#1F4E79; font-weight:600; font-size:1.1rem;'>
            👈 Choose a mode from the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)

def calculate_budget_forecast(
    state: str,
    spending: dict,
    monthly_salary: int,
    horizon_months: int,
) -> dict:
    """
    Orchestrates the full budget forecast:
    - Calls /forecast for each category the user has spending in
    - Computes the household-level summary: felt vs headline inflation,
      biggest leak/saving category, projected surplus
    """
    active_categories = {k: v for k, v in spending.items() if v > 0}

    if not active_categories:
        return None

    category_forecasts = {}
    total_current = 0
    total_projected_final_month = 0

    for div, amount in active_categories.items():
        result = fetch_forecast(state, div, horizon_months)
        if result is None:
            continue

        forecast_list = result["forecast"]
        first_index = forecast_list[0]["predicted_index"]
        final_index = forecast_list[-1]["predicted_index"]

        # Scale index change onto the user's actual RM amount for this category
        projected_amounts = [
            amount * (f["predicted_index"] / first_index) for f in forecast_list
        ]

        category_forecasts[div] = {
            "category_name": CATEGORIES.get(div, div),
            "current_rm": amount,
            "projected_final_rm": round(projected_amounts[-1], 2),
            "change_rm": round(projected_amounts[-1] - amount, 2),
            "change_pct": round((final_index - first_index) / first_index * 100, 2),
            "forecast_raw": forecast_list,
            "projected_amounts": projected_amounts,
        }

        total_current += amount
        total_projected_final_month += projected_amounts[-1]

    if not category_forecasts:
        return None

    # ── household-level "felt" inflation ────────────────────────────────────
    # Weighted average of category % changes, weighted by how much
    # the user actually spends in each — this IS their personal
    # inflation rate, just derived from RM inputs instead of % sliders
    felt_inflation_pct = round(
        (total_projected_final_month - total_current) / total_current * 100, 2
    )

    # ── leak / saving category — biggest mover in each direction ──────────────
    sorted_by_change = sorted(
        category_forecasts.items(),
        key=lambda x: x[1]["change_pct"],
        reverse=True,
    )
    biggest_leak = sorted_by_change[0]
    biggest_saving = sorted_by_change[-1]

    # ── surplus projection ──────────────────────────────────────────────────
    current_surplus = monthly_salary - total_current
    projected_surplus = monthly_salary - total_projected_final_month

    return {
        "category_forecasts": category_forecasts,
        "total_current": round(total_current, 2),
        "total_projected_final": round(total_projected_final_month, 2),
        "felt_inflation_pct": felt_inflation_pct,
        "biggest_leak": {
            "name": biggest_leak[1]["category_name"],
            "change_pct": biggest_leak[1]["change_pct"],
        },
        "biggest_saving": {
            "name": biggest_saving[1]["category_name"],
            "change_pct": biggest_saving[1]["change_pct"],
        },
        "current_surplus": round(current_surplus, 2),
        "projected_surplus": round(projected_surplus, 2),
        "horizon_months": horizon_months,
    }

# ── personal calculator page ───────────────────────────────────────

def render_personal_calculator(
    state: str,
    spending: dict,
    monthly_salary: int,
    horizon_months: int,
):
    """
    Budget Forecast page — replaces the old % based calculator.
    Shows RM-based projections, felt vs headline inflation,
    leak/saving categories, and surplus forecast.
    """
    st.title("📊 Your Budget Forecast")
    st.caption(
        f"Based on your actual spending in {state} · "
        f"{horizon_months}-month forecast · Latest data: April 2026"
    )

    total_spent = sum(spending.values())

    if total_spent == 0:
        st.info("Enter your spending per category in the sidebar to get started.")
        st.stop()

    if monthly_salary > 0 and total_spent > monthly_salary:
        st.warning(
            "Your spending exceeds your salary — the forecast will still run, "
            "but check your numbers in the sidebar."
        )

    with st.spinner(f"Forecasting {horizon_months} months ahead..."):
        result = calculate_budget_forecast(state, spending, monthly_salary, horizon_months)

    if result is None:
        st.error("Could not generate a forecast. Please try again.")
        st.stop()

    # ── the validation sentence — leads the page ────────────────────────────
    felt = result["felt_inflation_pct"]
    headline_pct = 1.5  # TODO: pull this live from a national headline endpoint

    direction_word = "higher" if felt > headline_pct else "lower"
    validation_color = "#dc2626" if felt > headline_pct else "#16a34a"

    st.markdown(f"""
    <div class="summary-box" style="border-left-color:{validation_color}">
        <strong>📢 The reality check</strong><br><br>
        Official inflation is currently around <b>{headline_pct}%</b>.
        Based on <b>your actual spending</b>, your personal inflation
        rate over the next {horizon_months} months is projected at
        <b style="color:{validation_color}">{felt:+.2f}%</b> —
        {abs(felt - headline_pct):.1f} points {direction_word} than the
        official number.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── top metrics row ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current monthly spending",
            f"RM{result['total_current']:,.0f}",
        )
    with col2:
        delta_rm = result["total_projected_final"] - result["total_current"]
        st.metric(
            f"Projected spending (month {horizon_months})",
            f"RM{result['total_projected_final']:,.0f}",
            delta=f"{delta_rm:+,.0f} RM",
            delta_color="inverse",
        )
    with col3:
        surplus_delta = result["projected_surplus"] - result["current_surplus"]
        st.metric(
            "Projected monthly surplus",
            f"RM{result['projected_surplus']:,.0f}",
            delta=f"{surplus_delta:+,.0f} RM",
            delta_color="normal",
        )

    st.divider()

    # ── leak / saving callouts ────────────────────────────────────────────────
    col_leak, col_save = st.columns(2)

    with col_leak:
        leak = result["biggest_leak"]
        st.markdown(f"""
        <div style='background:#FEF2F2; border-left:4px solid #dc2626;
                    border-radius:8px; padding:16px 20px;'>
            <b style='color:#991B1B'>📈 Your biggest leak</b><br><br>
            <b>{leak['name']}</b><br>
            <span style='color:#7F1D1D'>
            Rising {leak['change_pct']:+.1f}% over {horizon_months} months —
            watch this category closely.
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_save:
        save = result["biggest_saving"]
        save_word = "falling" if save["change_pct"] < 0 else "rising slowest"
        st.markdown(f"""
        <div style='background:#F0FDF4; border-left:4px solid #16a34a;
                    border-radius:8px; padding:16px 20px;'>
            <b style='color:#166534'>💚 Your best category</b><br><br>
            <b>{save['name']}</b><br>
            <span style='color:#14532D'>
            {save_word.capitalize()} ({save['change_pct']:+.1f}%) —
            this is where you're least exposed.
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── per-category forecast charts ────────────────────────────────────────
    st.subheader("Category forecasts")
    st.caption("How much you'll spend per category, with a widening confidence range the further out we forecast.")

    sorted_cats = sorted(
        result["category_forecasts"].items(),
        key=lambda x: x[1]["current_rm"],
        reverse=True,
    )

    for div, cat_data in sorted_cats[:4]:  # show top 4 by spend to avoid clutter
        fig = render_forecast_chart(
            cat_data["forecast_raw"],
            cat_data["category_name"],
            base_amount_rm=cat_data["current_rm"],
        )
        st.plotly_chart(fig, use_container_width=True)

    if len(sorted_cats) > 4:
        with st.expander(f"Show {len(sorted_cats) - 4} more categories"):
            for div, cat_data in sorted_cats[4:]:
                fig = render_forecast_chart(
                    cat_data["forecast_raw"],
                    cat_data["category_name"],
                    base_amount_rm=cat_data["current_rm"],
                )
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── full category table ───────────────────────────────────────────────────
    st.subheader("Full breakdown")

    table_data = []
    for div, cat_data in sorted_cats:
        table_data.append({
            "Category": cat_data["category_name"],
            "Current (RM)": f"RM{cat_data['current_rm']:,.0f}",
            f"Projected month {horizon_months} (RM)": f"RM{cat_data['projected_final_rm']:,.0f}",
            "Change": f"{cat_data['change_rm']:+,.0f} RM ({cat_data['change_pct']:+.1f}%)",
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ Forecasts use recursive prediction — accuracy naturally decreases "
        "further into the horizon as the model builds on its own prior "
        "predictions. Treat the widening range as a guide, not a guarantee."
    )

# ── relocation planner page ───────────────────────────────────────────────────

def render_relocation_planner(
    origin_state: str,
    spending: dict,
    monthly_salary: int,
):
    """
    Relocation Planner — RM-based comparison between two states.

    Auto-suggests destination spending by scaling origin RM amounts
    by the current price ratio between states, then lets the user
    adjust before running a full multi-month forecast comparison.
    """
    st.title("🏠 Relocation Cost Planner")
    st.caption(
        "Compare your real cost of living between two Malaysian states, "
        "in ringgit — using official DOSM and BNM data"
    )

    active_spending = {k: v for k, v in spending.items() if v > 0}

    if not active_spending:
        st.info("Enter your spending in the sidebar to get started.")
        st.stop()

    # ── destination state + horizon selectors ───────────────────────────────
    col_origin, col_arrow, col_dest = st.columns([2, 1, 2])

    with col_origin:
        st.info(f"**Moving from:** {origin_state}")

    with col_arrow:
        st.markdown(
            "<div style='text-align:center;font-size:2rem;padding-top:8px'>→</div>",
            unsafe_allow_html=True,
        )

    with col_dest:
        destination_state = st.selectbox(
            "Moving to",
            options=[s for s in STATES if s != origin_state],
            index=STATES.index("W.P. Kuala Lumpur")
            if origin_state != "W.P. Kuala Lumpur"
            else 0,
        )

    horizon_months = st.radio(
        "Forecast horizon",
        options=[3, 6, 12],
        index=1,
        format_func=lambda x: f"{x} months",
        horizontal=True,
        key="reloc_horizon",
    )

    st.divider()

    # ── fetch price ratios for every active category ────────────────────────
    st.subheader("Suggested spending in your new state")
    st.caption(
        f"Auto-scaled from your {origin_state} spending using current "
        f"{destination_state} prices. Adjust any amount if you expect "
        f"your habits to change."
    )

    destination_spending = {}
    ratio_info = {}

    with st.spinner(f"Comparing prices between {origin_state} and {destination_state}..."):
        for div, origin_amount in active_spending.items():
            ratio_data = fetch_price_ratio(origin_state, destination_state, div)

            if ratio_data:
                ratio = ratio_data["ratio"]
                pct_diff = ratio_data["pct_difference"]
            else:
                # Fallback if the ratio lookup fails for this category —
                # assume no price difference rather than blocking the page
                ratio = 1.0
                pct_diff = 0.0

            suggested_amount = round(origin_amount * ratio)
            ratio_info[div] = {"ratio": ratio, "pct_diff": pct_diff}

            destination_spending[div] = suggested_amount

    # ── editable destination table ───────────────────────────────────────────
    st.markdown("**Your destination budget (editable):**")

    edited_spending = {}
    for div, suggested in destination_spending.items():
        col_name, col_origin_amt, col_dest_amt, col_diff = st.columns([2.5, 1.3, 1.5, 1.2])

        with col_name:
            st.markdown(f"{CATEGORIES.get(div, div)}")
        with col_origin_amt:
            st.markdown(f"RM{active_spending[div]:,.0f}")
        with col_dest_amt:
            new_val = st.number_input(
                "dest",
                min_value=0,
                max_value=20000,
                value=int(suggested),
                step=25,
                key=f"dest_amt_{div}",
                label_visibility="collapsed",
            )
            edited_spending[div] = new_val
        with col_diff:
            pct = ratio_info[div]["pct_diff"]
            color = "#dc2626" if pct > 0 else "#16a34a"
            st.markdown(
                f"<span style='color:{color}'>{pct:+.1f}%</span>",
                unsafe_allow_html=True,
            )

    st.divider()
    # ── validate before running the full comparison ──────────────────────────
    total_dest = sum(edited_spending.values())
    if total_dest == 0:
        st.warning("Set at least one destination spending amount above.")
        st.stop()

    if not st.button("🔍 Compare full forecast", type="primary"):
        st.caption("Click above to run the full multi-month comparison.")
        st.stop()

    # ── run /forecast for BOTH states, every active category ─────────────────
    origin_results = {}
    dest_results = {}

    with st.spinner(f"Forecasting {horizon_months} months for both states..."):
        for div, amount in active_spending.items():
            origin_forecast = fetch_forecast(origin_state, div, horizon_months)
            if origin_forecast:
                origin_results[div] = origin_forecast

        for div, amount in edited_spending.items():
            if amount <= 0:
                continue
            dest_forecast = fetch_forecast(destination_state, div, horizon_months)
            if dest_forecast:
                dest_results[div] = dest_forecast

    if not origin_results or not dest_results:
        st.error("Could not generate forecasts. Please try again.")
        st.stop()

    # ── compute RM projections for both states ────────────────────────────────
    def project_rm(forecast_result, base_amount):
        forecast_list = forecast_result["forecast"]
        first_index = forecast_list[0]["predicted_index"]
        final_index = forecast_list[-1]["predicted_index"]
        projected_amounts = [
            base_amount * (f["predicted_index"] / first_index) for f in forecast_list
        ]
        return projected_amounts, forecast_list

    origin_total_current = sum(active_spending.values())
    origin_total_final = 0
    dest_total_current = sum(edited_spending.values())
    dest_total_final = 0

    category_comparisons = {}

    for div in origin_results:
        if div not in dest_results:
            continue

        origin_amounts, origin_raw = project_rm(origin_results[div], active_spending[div])
        dest_amounts, dest_raw = project_rm(dest_results[div], edited_spending[div])

        origin_total_final += origin_amounts[-1]
        dest_total_final += dest_amounts[-1]

        category_comparisons[div] = {
            "category_name": CATEGORIES.get(div, div),
            "origin_amounts": origin_amounts,
            "dest_amounts": dest_amounts,
            "origin_raw": origin_raw,
            "dest_raw": dest_raw,
        }

    # ── summary banner ────────────────────────────────────────────────────────
    cost_diff_rm = dest_total_final - origin_total_final
    cost_diff_pct = (cost_diff_rm / origin_total_final * 100) if origin_total_final else 0
    salary_needed = round(dest_total_final * 1.30)

    direction = "more" if cost_diff_rm > 0 else "less"
    st.markdown(f"""
    <div class="summary-box">
        <strong>📊 Relocation Summary</strong><br><br>
        By month {horizon_months}, your monthly spending in
        <b>{destination_state}</b> is projected at
        <b>RM{dest_total_final:,.0f}</b> — that's
        <b>RM{abs(cost_diff_rm):,.0f} {direction}</b>
        ({cost_diff_pct:+.1f}%) than staying in <b>{origin_state}</b>
        (projected RM{origin_total_final:,.0f}).<br><br>
        Recommended minimum salary in {destination_state}:
        <b>RM{salary_needed:,.0f}/month</b> (includes a 30% buffer).
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── side by side metrics ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            f"{origin_state} — projected (month {horizon_months})",
            f"RM{origin_total_final:,.0f}",
            delta=f"{origin_total_final - origin_total_current:+,.0f} RM",
            delta_color="inverse",
        )
    with col2:
        st.metric(
            f"{destination_state} — projected (month {horizon_months})",
            f"RM{dest_total_final:,.0f}",
            delta=f"{dest_total_final - dest_total_current:+,.0f} RM",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            "Extra monthly cost to relocate",
            f"RM{cost_diff_rm:,.0f}",
            delta=f"{cost_diff_pct:+.1f}%",
            delta_color="inverse",
        )

    st.divider()

    # ── per-category comparison charts ────────────────────────────────────────
    st.subheader("Category-by-category comparison")

    sorted_comparisons = sorted(
        category_comparisons.items(),
        key=lambda x: active_spending.get(x[0], 0),
        reverse=True,
    )

    for div, comp in sorted_comparisons[:4]:
        steps = [f"Month {i+1}" for i in range(len(comp["origin_amounts"]))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps, y=comp["origin_amounts"],
            mode="lines+markers", name=origin_state,
            line=dict(color="#64748b", width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=steps, y=comp["dest_amounts"],
            mode="lines+markers", name=destination_state,
            line=dict(color="#1F4E79", width=3),
        ))
        fig.update_layout(
            title=comp["category_name"],
            yaxis_title="Projected spending (RM)",
            height=320,
            margin=dict(t=50, b=30, l=50, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    if len(sorted_comparisons) > 4:
        with st.expander(f"Show {len(sorted_comparisons) - 4} more categories"):
            for div, comp in sorted_comparisons[4:]:
                steps = [f"Month {i+1}" for i in range(len(comp["origin_amounts"]))]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=steps, y=comp["origin_amounts"],
                    mode="lines+markers", name=origin_state,
                    line=dict(color="#64748b", width=2, dash="dot"),
                ))
                fig.add_trace(go.Scatter(
                    x=steps, y=comp["dest_amounts"],
                    mode="lines+markers", name=destination_state,
                    line=dict(color="#1F4E79", width=3),
                ))
                fig.update_layout(
                    title=comp["category_name"],
                    yaxis_title="Projected spending (RM)",
                    height=320,
                    margin=dict(t=50, b=30, l=50, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "⚠️ Destination amounts are auto-suggested by scaling your current "
        "spending using today's price ratio between states, then forecast "
        "forward independently for each state. Forecasts use recursive "
        "prediction — treat the further-out months as directional, not exact."
    )

# ── main entry point ──────────────────────────────────────────────────────────

def main():
    state, spending, monthly_salary, horizon_months, mode = render_sidebar()

    if mode == "🏠 Home":
        render_home()
    elif mode == "📊 Personal Inflation Calculator":
        render_personal_calculator(state, spending, monthly_salary, horizon_months)
    else:
        render_relocation_planner(state, spending, monthly_salary)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#94a3b8;font-size:0.8rem;padding:10px'>
        Malaysia Personal Inflation Intelligence &nbsp;|&nbsp;
        Data: DOSM & BNM via data.gov.my &nbsp;|&nbsp;
        Model: XGBoost trained on 2010–2026 CPI data &nbsp;|&nbsp;
        Built by Muhammad Ammar Ahmad Zaki
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()