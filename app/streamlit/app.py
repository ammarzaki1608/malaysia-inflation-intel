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

# ── page config — must be first Streamlit call ───────────────────────────────
st.set_page_config(
    page_title="Malaysia Inflation Intelligence",
    page_icon="🇲🇾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8080"

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
        options=["Personal Inflation Calculator", "Relocation Planner"],
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

    # Profile preset selector
    st.sidebar.subheader("Your spending profile")
    st.sidebar.caption("How do you distribute your monthly spending?")

    preset = st.sidebar.selectbox(
        "Start from a preset",
        options=list(DEFAULT_PROFILES.keys()),
        index=0,
    )

    # Load preset weights
    if preset != "Custom":
        base_weights = DEFAULT_PROFILES[preset].copy()
    else:
        base_weights = {div: 10 for div in list(CATEGORIES.keys())[:10]}

    # Sliders for each category
    st.sidebar.markdown("**Adjust your spending percentages:**")
    weights = {}
    total   = 0

    for div, name in CATEGORIES.items():
        default_val = base_weights.get(div, 0)
        val = st.sidebar.slider(
            f"{name}",
            min_value=0,
            max_value=60,
            value=default_val,
            step=1,
            key=f"slider_{div}",
        )
        weights[div] = val
        total += val

    # Show total with colour feedback
    if abs(total - 100) < 1:
        st.sidebar.success(f"Total: {total}% ✓")
    elif total > 100:
        st.sidebar.error(f"Total: {total}% — reduce by {total - 100}%")
    else:
        st.sidebar.warning(f"Total: {total}% — add {100 - total}% more")

    return state, weights, mode, total

# ── personal inflation calculator page ───────────────────────────────────────

def render_personal_calculator(state: str, weights: dict, total: int):
    """
    Main page for personal inflation calculator mode.
    Shows gauge, category breakdown, SHAP explanation,
    and income context.
    """
    st.title("Your Personal Inflation Rate")
    st.caption(
        f"Based on official DOSM data · Latest data: April 2026 · State: {state}"
    )

    # Validate weights before calling API
    if abs(total - 100) > 1:
        st.warning(
            f"Your spending weights add up to {total}%. "
            "Please adjust the sliders in the sidebar to reach exactly 100% "
            "before calculating."
        )
        st.stop()

    # Filter zero weights and convert keys to strings
    active_weights = weights_to_str_keys(
        {k: v for k, v in weights.items() if v > 0}
    )

    if not active_weights:
        st.info("Set your spending percentages in the sidebar to get started.")
        st.stop()

    # Call API
    with st.spinner("Calculating your personal inflation rate..."):
        result = fetch_prediction(state, active_weights)

    if not result:
        st.stop()

    # ── top metrics row ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    personal_rate = result["personal_inflation_rate"]
    headline_rate = result["headline_cpi_change"]
    diff          = personal_rate - headline_rate
    income        = result.get("income_context", {})

    with col1:
        fig = render_inflation_gauge(personal_rate, "Your personal inflation")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = render_inflation_gauge(headline_rate, "State average inflation")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        color = "positive" if diff > 0 else "negative"
        direction = "above" if diff > 0 else "below"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {color}">{diff:+.2f}%</div>
            <div class="metric-label">You are {abs(diff):.2f}%
            {direction} the state average</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        median = income.get("income_median")
        band   = income.get("income_band", "N/A")
        exp    = income.get("expenditure_mean")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem">
                RM{median:,.0f}
            </div>
            <div class="metric-label">
                {state} median income<br>
                Income band: {band}<br>
                Avg expenditure: RM{exp:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── spending profile + category breakdown ─────────────────────────────────
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Your spending profile")
        fig = render_spending_profile_pie(weights)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Inflation by category")
        fig = render_category_waterfall(result["category_breakdown"])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── category detail table ─────────────────────────────────────────────────
    st.subheader("Category breakdown")
    st.caption(
        "How much each spending category contributes to your "
        "personal inflation rate"
    )

    table_data = []
    for cat in result["category_breakdown"]:
        table_data.append({
            "Category":           cat["category_name"],
            "Your weight":        f"{cat['weight']:.0f}%",
            "Current index":      f"{cat['current_index']:.1f}",
            "Forecast index":     f"{cat['predicted_index']:.2f}",
            "Price change":       f"{cat['pct_change']:+.2f}%",
            "Your contribution":  f"{cat['weighted_contribution']:+.3f}%",
        })

    df = pd.DataFrame(table_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price change": st.column_config.TextColumn(
                "Price change",
                help="Forecasted % change in this category's CPI index"
            ),
            "Your contribution": st.column_config.TextColumn(
                "Your contribution",
                help="Price change × your spending weight"
            ),
        }
    )

    st.divider()

    # ── SHAP explanation ──────────────────────────────────────────────────────
    st.subheader("What is driving your inflation forecast?")
    st.caption(
        "SHAP values show which economic factors most influenced "
        "this prediction. Higher = more influential."
    )

    # Use category-level SHAP aggregated across top categories
    shap_agg: dict = {}
    for cat in result["category_breakdown"]:
        for sv in cat.get("shap_values", []):
            feat = sv["feature"]
            shap_agg[feat] = shap_agg.get(feat, 0.0) + abs(sv["value"])

    shap_list = sorted(
        [{"feature": k, "importance": round(v, 2)}
         for k, v in shap_agg.items()],
        key=lambda x: x["importance"],
        reverse=True,
    )[:5]

    col_shap, col_explain = st.columns([1, 1])

    with col_shap:
        fig = render_shap_bar(shap_list, "Top 5 forecast drivers")
        st.plotly_chart(fig, use_container_width=True)

    with col_explain:
        st.markdown("**What these features mean:**")
        explanations = {
            "cpi_ma_3":   "The 3-month CPI trend is the strongest signal — if prices have been rising steadily for 3 months, they are likely to continue rising.",
            "cpi_lag_1":  "Last month's CPI value is a strong predictor — inflation is sticky and rarely reverses sharply in a single month.",
            "cpi_lag_3":  "The CPI value 3 months ago captures medium-term momentum — whether inflation has been building or easing.",
            "ppi_lag_3":  "Producer prices from 3 months ago predict consumer prices today — factories pass cost increases to consumers with a lag.",
            "ppi_index":  "Current producer prices signal near-term consumer price direction.",
            "opr_rate":   "BNM's interest rate influences borrowing costs and consumer spending, which dampens or accelerates inflation.",
            "cpi_lag_12": "The CPI value 12 months ago captures seasonal patterns — food prices rise before Hari Raya every year.",
            "cpi_ma_12":  "The 12-month trend shows structural inflation direction.",
        }

        for item in shap_list:
            feat    = item["feature"]
            explain = explanations.get(feat, "Economic indicator influencing the forecast.")
            st.markdown(f"**{feat}** — {explain}")
            st.markdown("")

    st.divider()

    # ── income context ────────────────────────────────────────────────────────
    st.subheader(f"Income context — {state}")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric(
            "Median household income",
            f"RM{income.get('income_median', 0):,.0f}/mo"
        )
    with col_b:
        st.metric(
            "Mean expenditure",
            f"RM{income.get('expenditure_mean', 0):,.0f}/mo"
        )
    with col_c:
        gini = income.get("gini", 0)
        st.metric("Gini coefficient", f"{gini:.3f}")
    with col_d:
        poverty = income.get("poverty_rate", 0)
        st.metric("Poverty rate", f"{poverty:.1f}%")

    st.caption(
        "Income data from DOSM Household Income and Expenditure Survey (HIES) 2022. "
        "CPI forecasts based on XGBoost model trained on DOSM data 2010–2026."
    )

# ── relocation planner page ───────────────────────────────────────────────────

def render_relocation_planner(origin_state: str, weights: dict, total: int):
    """
    Relocation planner page — compares personal inflation and cost
    of living between two Malaysian states.
    """
    st.title("🏠 Relocation Cost Planner")
    st.caption(
        "Compare your cost of living between two Malaysian states "
        "using official DOSM and BNM data"
    )

    # Destination state selector
    col_origin, col_arrow, col_dest = st.columns([2, 1, 2])

    with col_origin:
        st.info(f"**Origin:** {origin_state}")

    with col_arrow:
        st.markdown(
            "<div style='text-align:center;font-size:2rem;padding-top:8px'>→</div>",
            unsafe_allow_html=True,
        )

    with col_dest:
        destination_state = st.selectbox(
            "Destination state",
            options=[s for s in STATES if s != origin_state],
            index=STATES.index("W.P. Kuala Lumpur")
            if origin_state != "W.P. Kuala Lumpur"
            else 0,
        )

    st.divider()

    # Destination spending profile
    st.subheader("Expected spending pattern in destination")
    st.caption(
        "Your spending habits will likely change when you move. "
        "Adjust the sliders below to reflect your expected lifestyle "
        f"in {destination_state}."
    )

    dest_preset = st.selectbox(
        "Start from a preset for destination",
        options=list(DEFAULT_PROFILES.keys()),
        index=2,
        key="dest_preset",
    )

    if dest_preset != "Custom":
        dest_base = DEFAULT_PROFILES[dest_preset].copy()
    else:
        dest_base = {div: 10 for div in list(CATEGORIES.keys())[:10]}

    # Destination sliders — two columns
    dest_weights = {}
    dest_total   = 0

    cols = st.columns(2)
    for i, (div, name) in enumerate(CATEGORIES.items()):
        with cols[i % 2]:
            default_val = dest_base.get(div, 0)
            val = st.slider(
                f"{name}",
                min_value=0,
                max_value=60,
                value=default_val,
                step=1,
                key=f"dest_slider_{div}",
            )
            dest_weights[div] = val
            dest_total += val

    if abs(dest_total - 100) < 1:
        st.success(f"Destination total: {dest_total}% ✓")
    elif dest_total > 100:
        st.error(f"Destination total: {dest_total}% — reduce by {dest_total - 100}%")
    else:
        st.warning(f"Destination total: {dest_total}% — add {100 - dest_total}% more")

    st.divider()

    # Validate before API call
    if abs(total - 100) > 1:
        st.warning(
            f"Your origin spending weights add up to {total}%. "
            "Please fix the sidebar sliders first."
        )
        st.stop()

    if abs(dest_total - 100) > 1:
        st.warning(
            f"Your destination spending weights add up to {dest_total}%. "
            "Please adjust the destination sliders above."
        )
        st.stop()

    # Call comparison API
    origin_weights = weights_to_str_keys(
        {k: v for k, v in weights.items() if v > 0}
    )
    dest_weights_clean = weights_to_str_keys(
        {k: v for k, v in dest_weights.items() if v > 0}
    )

    with st.spinner(
        f"Comparing {origin_state} vs {destination_state}..."
    ):
        result = fetch_comparison(
            origin_state,
            destination_state,
            origin_weights,
            dest_weights_clean,
        )

    if not result:
        st.stop()

    # ── summary banner ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="summary-box">
        <strong>📊 Relocation Summary</strong><br><br>
        {result['summary']}
    </div>
    """, unsafe_allow_html=True)

    # ── key metrics comparison ────────────────────────────────────────────────
    st.subheader("Side-by-side comparison")

    col1, col2, col3 = st.columns(3)

    origin_data = result["origin"]
    dest_data   = result["destination"]
    cost_diff   = result["cost_difference_pct"]
    salary      = result["salary_needed"]

    with col1:
        st.metric(
            f"Personal inflation — {origin_state}",
            f"{origin_data['personal_inflation_rate']:+.2f}%",
        )
        st.metric(
            f"Median income — {origin_state}",
            f"RM{origin_data['income_context'].get('income_median', 0):,.0f}",
        )
        st.metric(
            f"Mean expenditure — {origin_state}",
            f"RM{origin_data['income_context'].get('expenditure_mean', 0):,.0f}",
        )
        st.metric(
            f"Income band — {origin_state}",
            origin_data["income_context"].get("income_band", "N/A"),
        )

    with col2:
        direction  = "higher" if cost_diff > 0 else "lower"
        diff_color = "red" if cost_diff > 0 else "green"
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:8px">
                <div style="font-size:3rem">
                    {"⬆️" if cost_diff > 0 else "⬇️"}
                </div>
                <div class="metric-value" style="color:{diff_color}">
                    {abs(cost_diff):.2f}%
                </div>
                <div class="metric-label">
                    inflation rate is {direction}<br>
                    in {destination_state}
                </div>
                <br>
                <div class="metric-value" style="font-size:1.6rem">
                    RM{salary:,.0f}
                </div>
                <div class="metric-label">
                    recommended minimum<br>
                    monthly salary in {destination_state}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.metric(
            f"Personal inflation — {destination_state}",
            f"{dest_data['personal_inflation_rate']:+.2f}%",
        )
        st.metric(
            f"Median income — {destination_state}",
            f"RM{dest_data['income_context'].get('income_median', 0):,.0f}",
        )
        st.metric(
            f"Mean expenditure — {destination_state}",
            f"RM{dest_data['income_context'].get('expenditure_mean', 0):,.0f}",
        )
        st.metric(
            f"Income band — {destination_state}",
            dest_data["income_context"].get("income_band", "N/A"),
        )

    st.divider()

    # ── gauge comparison ──────────────────────────────────────────────────────
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig = render_inflation_gauge(
            origin_data["personal_inflation_rate"],
            f"Personal inflation — {origin_state}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_g2:
        fig = render_inflation_gauge(
            dest_data["personal_inflation_rate"],
            f"Personal inflation — {destination_state}",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── category price comparison ─────────────────────────────────────────────
    st.subheader("Price level comparison by category")
    st.caption(
        "Current CPI index values by category — higher index means "
        "historically more expensive relative to the 2010 base year"
    )

    origin_cats = {
        c["division"]: c for c in origin_data["category_breakdown"]
    }
    dest_cats = {
        c["division"]: c for c in dest_data["category_breakdown"]
    }

    all_divs = sorted(
        set(origin_cats.keys()) | set(dest_cats.keys())
    )

    comparison_rows = []
    for div in all_divs:
        o = origin_cats.get(div, {})
        d = dest_cats.get(div, {})
        if not o or not d:
            continue

        o_idx = o.get("current_index", 0)
        d_idx = d.get("current_index", 0)
        diff  = ((d_idx - o_idx) / o_idx * 100) if o_idx else 0

        comparison_rows.append({
            "Category":              o.get("category_name", div),
            f"{origin_state} index": f"{o_idx:.1f}",
            f"{destination_state} index": f"{d_idx:.1f}",
            "Difference":            f"{diff:+.1f}%",
            "More expensive in":     destination_state if diff > 0 else origin_state,
        })

    df_compare = pd.DataFrame(comparison_rows)
    st.dataframe(
        df_compare,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── waterfall charts side by side ─────────────────────────────────────────
    st.subheader("Inflation contribution by category")

    col_wf1, col_wf2 = st.columns(2)

    with col_wf1:
        st.markdown(f"**{origin_state}**")
        fig = render_category_waterfall(origin_data["category_breakdown"])
        st.plotly_chart(fig, use_container_width=True)

    with col_wf2:
        st.markdown(f"**{destination_state}**")
        fig = render_category_waterfall(dest_data["category_breakdown"])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── relocation checklist ──────────────────────────────────────────────────
    st.subheader(f"Things to know before moving to {destination_state}")

    dest_income = dest_data["income_context"]
    origin_income = origin_data["income_context"]

    dest_exp    = dest_income.get("expenditure_mean", 0) or 0
    origin_exp  = origin_income.get("expenditure_mean", 0) or 0
    exp_ratio   = (dest_exp / origin_exp) if origin_exp > 0 else 1

    st.markdown(f"""
    - **Cost of living multiplier:** spending in {destination_state} is
      approximately **{exp_ratio:.1f}x** the cost of {origin_state}
      based on mean household expenditure data
    - **Income band difference:** {origin_state} is predominantly
      **{origin_income.get('income_band', 'N/A')}** while
      {destination_state} is predominantly
      **{dest_income.get('income_band', 'N/A')}**
    - **Gini coefficient:** {origin_state} {origin_income.get('gini', 0):.3f}
      vs {destination_state} {dest_income.get('gini', 0):.3f}
      — {'higher inequality in destination' if dest_income.get('gini',0) > origin_income.get('gini',0) else 'lower inequality in destination'}
    - **Poverty rate:** {origin_state} {origin_income.get('poverty_rate', 0):.1f}%
      vs {destination_state} {dest_income.get('poverty_rate', 0):.1f}%
    - **Recommended salary:** at least **RM{salary:,.0f}/month** to cover
      destination expenditure with a 30% savings buffer
    """)

    st.caption(
        "⚠️ This tool uses state-level averages from DOSM HIES 2022 and "
        "CPI data to April 2026. Actual costs vary by neighbourhood, "
        "lifestyle, and individual circumstances. Use this as a planning "
        "guide, not a guarantee."
    )

# ── main entry point ──────────────────────────────────────────────────────────

def main():
    """Main app entry point — routes between modes."""

    # Render sidebar and get user inputs
    state, weights, mode, total = render_sidebar()

    # Route to correct page based on mode
    if mode == "Personal Inflation Calculator":
        render_personal_calculator(state, weights, total)
    else:
        render_relocation_planner(state, weights, total)

    # ── footer ────────────────────────────────────────────────────────────────
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