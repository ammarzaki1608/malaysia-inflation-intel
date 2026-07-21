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

LANG = {
    "en": {
        "app_title": "Malaysia Personal Inflation Intelligence",
        "app_tagline": "Built on official DOSM & BNM data",
        "mode_label": "Choose mode",
        "mode_home": "🏠 Home",
        "mode_calculator": "📊 Personal Inflation Calculator",
        "mode_relocation": "🗺️ Relocation Planner",
        "origin_state_label": "Origin state (where you live now)",
        "budget_header": "💰 Your monthly budget",
        "budget_caption": "Enter real amounts — we'll do the maths.",
        "salary_label": "Monthly salary (RM)",
        "spending_header": "Spending per category (RM):",
        "allocated_label": "Allocated",
        "unallocated_label": "Unallocated (savings)",
        "over_budget_error": "You've allocated RM{amount:,} more than your salary.",
        "horizon_label": "Forecast horizon",
        "budget_forecast_title": "📊 Your Budget Forecast",
        "reality_check_label": "📢 The reality check",
        "current_spending_label": "Current monthly spending",
        "projected_spending_label": "Projected spending (month {n})",
        "projected_surplus_label": "Projected monthly surplus",
        "biggest_leak_label": "📈 Your biggest leak",
        "best_category_label": "💚 Your best category",
        "category_forecasts_header": "Category forecasts",
        "full_breakdown_header": "Full breakdown",
        "relocation_title": "🏠 Relocation Cost Planner",
        "moving_from_label": "Moving from",
        "moving_to_label": "Moving to",
        "suggested_spending_header": "Suggested spending in your new state",
        "compare_button": "🔍 Compare full forecast",
        "salary_needed_label": "Recommended minimum salary",
        "mode_explorer": "📈 Data Explorer",
        "explorer_title": "📈 Data Explorer",
        "explorer_caption": "See the real historical trends behind your forecast — CPI, producer prices, and interest rates, straight from DOSM and BNM.",
        "explorer_state_label": "State",
        "explorer_category_label": "Category",
        "explorer_range_label": "Time range",
        "cpi_trend_header": "CPI Trend",
        "ppi_overlay_header": "Producer Prices vs Consumer Prices",
        "opr_overlay_header": "Interest Rate vs Inflation",
        "income_header": "Income & Expenditure Snapshot (HIES 2022)",
        "mode_ask": "💬 Ask",
        "ask_title": "💬 Ask About Your Forecast",
        "ask_caption": "Ask a question in plain language about your inflation forecast — answers are grounded in your actual data, not generic advice.",
        "ask_state_label": "State",
        "ask_category_label": "Category",
        "ask_question_label": "Your question",
        "ask_question_placeholder": "e.g. Why does my food inflation forecast get less certain further out?",
        "ask_button": "Ask",
        "ask_thinking": "Thinking through your question...",
        "ask_error": "Could not get an answer right now. Please try again.",
        "ask_grounded_in": "This answer is based on your forecast for",
        "national_overview_header": "🇲🇾 The National Picture",
        "national_overview_caption": "Here's how inflation looks across Malaysia as a whole, before we drill down.",
        "category_leaderboard_header": "Which categories are rising fastest?",
        "category_leaderboard_caption": "Year-on-year change, nationally, across all 13 CPI categories.",
        "state_leaderboard_header": "How does this category compare across states?",
        "state_leaderboard_caption": "Current index by state for the category selected below.",
        "explore_further_text": "Want to see how this plays out over time in your state? Explore below.",
    },
    "ms": {
        "app_title": "Maklumat Inflasi Peribadi Malaysia",
        "app_tagline": "Dibangunkan berdasarkan data rasmi DOSM & BNM",
        "mode_label": "Pilih mod",
        "mode_home": "🏠 Laman Utama",
        "mode_calculator": "📊 Kalkulator Inflasi Peribadi",
        "mode_relocation": "🗺️ Perancang Perpindahan",
        "origin_state_label": "Negeri asal (tempat kediaman semasa)",
        "budget_header": "💰 Belanjawan bulanan anda",
        "budget_caption": "Masukkan jumlah sebenar — kami akan membuat pengiraan.",
        "salary_label": "Gaji bulanan (RM)",
        "spending_header": "Perbelanjaan mengikut kategori (RM):",
        "allocated_label": "Diperuntukkan",
        "unallocated_label": "Tidak diperuntukkan (simpanan)",
        "over_budget_error": "Anda telah memperuntukkan RM{amount:,} lebih daripada gaji anda.",
        "horizon_label": "Tempoh unjuran",
        "budget_forecast_title": "📊 Unjuran Belanjawan Anda",
        "reality_check_label": "📢 Semakan realiti",
        "current_spending_label": "Perbelanjaan bulanan semasa",
        "projected_spending_label": "Unjuran perbelanjaan (bulan ke-{n})",
        "projected_surplus_label": "Unjuran lebihan bulanan",
        "biggest_leak_label": "📈 Ketirisan belanjawan terbesar",
        "best_category_label": "💚 Kategori paling menjimatkan",
        "category_forecasts_header": "Unjuran kategori",
        "full_breakdown_header": "Pecahan terperinci",
        "relocation_title": "🏠 Perancang Kos Perpindahan",
        "moving_from_label": "Berpindah dari",
        "moving_to_label": "Berpindah ke",
        "suggested_spending_header": "Cadangan perbelanjaan di negeri baharu anda",
        "compare_button": "🔍 Bandingkan unjuran penuh",
        "salary_needed_label": "Gaji minimum yang disyorkan",
        "mode_explorer": "📈 Peneroka Data",
        "explorer_title": "📈 Peneroka Data",
        "explorer_caption": "Lihat trend sejarah sebenar di sebalik unjuran anda — CPI, harga pengeluar, dan kadar faedah, terus daripada DOSM dan BNM.",
        "explorer_state_label": "Negeri",
        "explorer_category_label": "Kategori",
        "explorer_range_label": "Julat masa",
        "cpi_trend_header": "Trend CPI",
        "ppi_overlay_header": "Harga Pengeluar lwn Harga Pengguna",
        "opr_overlay_header": "Kadar Faedah lwn Inflasi",
        "income_header": "Ringkasan Pendapatan & Perbelanjaan (HIES 2022)",
        "mode_ask": "💬 Tanya",
        "ask_title": "💬 Tanya Tentang Unjuran Anda",
        "ask_caption": "Tanya soalan dalam bahasa mudah tentang unjuran inflasi anda — jawapan yang diberikan adalah berlandaskan data sebenar anda, dan bukannya sekadar nasihat umum.",
        "ask_state_label": "Negeri",
        "ask_category_label": "Kategori",
        "ask_question_label": "Soalan anda",
        "ask_question_placeholder": "cth. Mengapakah unjuran inflasi makanan saya menjadi kurang pasti apabila tempoh unjuran semakin jauh?",
        "ask_button": "Tanya",
        "ask_thinking": "Sedang memproses soalan anda...",
        "ask_error": "Tidak dapat menjana jawapan pada masa ini. Sila cuba lagi.",
        "ask_grounded_in": "Jawapan ini adalah berdasarkan unjuran anda untuk",
        "national_overview_header": "🇲🇾 Gambaran Kebangsaan",
        "national_overview_caption": "Berikut adalah gambaran inflasi Malaysia secara keseluruhan, sebelum kita mendalami butiran.",
        "category_leaderboard_header": "Kategori mana yang meningkat paling pantas?",
        "category_leaderboard_caption": "Perubahan tahun ke tahun, secara kebangsaan, merentasi semua 13 kategori CPI.",
        "state_leaderboard_header": "Bagaimana kategori ini dibandingkan antara negeri?",
        "state_leaderboard_caption": "Indeks semasa mengikut negeri bagi kategori yang dipilih di bawah.",
        "explore_further_text": "Nak lihat bagaimana ia berlaku dari semasa ke semasa di negeri anda? Terokai di bawah.",
    }
}

CATEGORY_NAMES_MS = {
    "01": "Makanan & Minuman Tanpa Alkohol",
    "02": "Minuman Beralkohol & Tembakau",
    "03": "Pakaian & Kasut",
    "04": "Perumahan, Air, Elektrik & Gas",
    "05": "Perabot & Peralatan Rumah",
    "06": "Kesihatan",
    "07": "Pengangkutan",
    "08": "Maklumat & Komunikasi",
    "09": "Rekreasi, Sukan & Kebudayaan",
    "10": "Pendidikan",
    "11": "Restoran & Penginapan",
    "12": "Insurans & Perkhidmatan Kewangan",
    "13": "Penjagaan Diri & Pelbagai",
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
    
@st.cache_data(ttl=3600)
def fetch_headline_inflation() -> dict:
    """
    Get the current national headline CPI year-on-year change.
    Cached for 1 hour since this changes at most once a month.
    """
    try:
        response = httpx.get(f"{API_URL}/headline-inflation", timeout=15.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[WARNING] fetch_headline_inflation failed: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_trends(state: str, division: str, months_back: int = 60) -> dict:
    """Get raw historical CPI/PPR/OPR time series for charting."""
    try:
        response = httpx.get(
            f"{API_URL}/trends",
            params={"state": state, "division": division, "months_back": months_back},
            timeout=20.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[WARNING] fetch_trends failed: {e}")
        return None
    
@st.cache_data(ttl=3600)
def fetch_income_snapshot(state: str) -> dict:
    """Get the HIES 2022 income/expenditure snapshot for one state."""
    try:
        response = httpx.get(
            f"{API_URL}/income-snapshot",
            params={"state": state},
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[WARNING] fetch_income_snapshot failed: {e}")
        return None

def fetch_ask_answer(question: str, state: str, division: str | None, horizon_months: int) -> dict:
    try:
        payload = {"question": question, "horizon_months": horizon_months}
        if state:
            payload["state"] = state
        if division:
            payload["division"] = division

        response = httpx.post(f"{API_URL}/ask", json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        return {"error": "timeout"}
    except Exception as e:
        print(f"[WARNING] fetch_ask_answer failed: {e}")
        return {"error": str(e)}
    
def fetch_prescriptive_insight(context_type: str, summary_data: dict) -> dict:
    """
    Get an auto-generated, general budgeting suggestion grounded in
    the report's own numbers. Not cached — always regenerated with
    the report, since summary_data changes each time.
    """
    try:
        response = httpx.post(
            f"{API_URL}/prescriptive-insight",
            json={"context_type": context_type, "summary_data": summary_data},
            timeout=20.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[WARNING] fetch_prescriptive_insight failed: {e}")
        return None
    
@st.cache_data(ttl=3600)
def fetch_category_leaderboard() -> dict:
    try:
        response = httpx.get(f"{API_URL}/category-leaderboard", timeout=15.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[WARNING] fetch_category_leaderboard failed: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_state_leaderboard(division: str) -> dict:
    try:
        response = httpx.get(f"{API_URL}/state-leaderboard", params={"division": division}, timeout=15.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[WARNING] fetch_state_leaderboard failed: {e}")
        return None

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

# ── sidebar — spending profile builder ───────────────────────────────────────

def render_sidebar() -> tuple:
    """
    Renders the sidebar with state selector and spending profile sliders.
    Returns (state, weights, mode) where mode is 'personal' or 'relocation'.
    """
    lang_choice = st.sidebar.radio(
        "Language / Bahasa",
        options=["EN", "BM"],
        horizontal=True,
        key="lang_toggle",
    )
    lang = "en" if lang_choice == "EN" else "ms"
    t = LANG[lang]

    st.sidebar.markdown(f"### {t['app_title']}")
    st.sidebar.caption(t['app_tagline'])
    st.sidebar.divider()

    # Mode selector
    mode = st.sidebar.radio(
        t["mode_label"],
        options=[t["mode_home"], t["mode_explorer"], t["mode_calculator"], t["mode_relocation"]],
        key="mode_select",
    )
    st.sidebar.divider()

    # State selector
    state = st.sidebar.selectbox(t["origin_state_label"], options=STATES)

    st.sidebar.divider()

    # ── salary + RM spending inputs (replaces % sliders) ──────────────────────
    st.sidebar.divider()
    st.sidebar.subheader(t["budget_header"])
    st.sidebar.caption(t["budget_caption"])

    monthly_salary = st.sidebar.number_input(
        t["salary_label"],
        min_value=0,
        max_value=50000,
        value=4500,
        step=100,
    )

    st.sidebar.markdown(f"**{t['spending_header']}**")

    spending = {}
    total_spent = 0

    default_spending = {
        "01": 900, "04": 700, "07": 400, "11": 300,
        "10": 150, "06": 150, "03": 150, "09": 150,
    }

    # Category names switch language, but the dict KEYS ("01", "04"...)
    # stay identical — this is what keeps the API calls language-independent
    category_labels = CATEGORIES if lang == "en" else CATEGORY_NAMES_MS

    for div, name in category_labels.items():
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
    st.sidebar.metric(t["allocated_label"], f"RM{total_spent:,}")
    st.sidebar.metric(t["unallocated_label"], f"RM{unallocated:,}")

    if unallocated < 0:
        st.sidebar.error(t["over_budget_error"].format(amount=abs(unallocated)))

    st.sidebar.divider()

    horizon_months = st.sidebar.radio(
        t["horizon_label"],
        options=[3, 6, 12],
        index=1,
        format_func=lambda x: f"{x} months" if lang == "en" else f"{x} bulan",
        horizontal=True,
    )

    return state, spending, monthly_salary, horizon_months, mode, t, lang

def render_home(t, lang):
    """Landing page — explains what the app does and how to use it."""

    if lang == "en":
        hero_title = "Malaysia Personal Inflation Intelligence"
        hero_sub = "The government publishes one inflation number for the whole country. But your inflation depends on how you actually spend your money. This tool calculates yours."
        section1_title = "💡 What is personal inflation?"
        section1_body = (
            "Malaysia's official CPI is calculated using a fixed national basket — "
            "an assumed average of how Malaysians spend money across 13 categories.\n\n"
            "But if you spend 40% of your income on food and food prices rise 5%, "
            "you feel that much more than someone who spends only 20% on food.\n\n"
            "Your inflation = your spending pattern × actual price changes per category."
        )
        example_label = "Example"
        example_body = "Government headline CPI: 2.1%<br><br>Person A (heavy food spender): 2.8% 📈<br>Person B (heavy housing spender): 1.7% 📉<br><br>Same month. Same country. Different reality."
        section2_title = "🛠️ What you can do with this app"
        card1_title = "Budget Forecast"
        card1_body = "Enter your salary and how much you spend per category in ringgit. We project your spending 3, 6, or 12 months ahead, show you how it compares to official inflation, and flag which category is quietly eating into your budget the most."
        card1_best = "Best for: seeing what your money actually buys next year."
        card2_title = "Relocation Planner"
        card2_body = "Thinking of moving to KL, Johor, or another state? We auto-scale your current spending using real price differences between states, forecast both side by side in ringgit, and estimate the salary you'd need to keep your standard of living."
        card2_best = "Best for: evaluating a job offer in another state."
        cta_text = "👈 Choose a mode from the sidebar to get started."
        ready_text = "Ready to find out your personal inflation rate?"
        step1_title = "Pick your mode"
        step1_body = "Choose Personal Calculator or Relocation Planner from the sidebar on the left."
        step2_title = "Enter your real numbers"
        step2_body = "Type in your monthly salary and how much you actually spend per category in ringgit — no percentages, no guesswork. Pick how far ahead you want to forecast: 3, 6, or 12 months."
        step3_title = "Read your results"
        step3_body = "See how your spending compares to official inflation, which category is your biggest leak, and what your monthly surplus looks like months from now."
        howto_header = "📖 How to use this app — 3 steps"
        data_header = "📡 Where does the data come from?"
        data_intro = "All data is sourced from official Malaysian government APIs — no third-party or estimated data. Updated monthly."
        disclaimer_header = "⚠️ Important disclaimer — please read"
        dosm_body = "Department of Statistics Malaysia — monthly CPI by state and category, Producer Price Index, and household income survey data."
        bnm_body = "Monthly interest rate data including the Overnight Policy Rate (OPR) — used as an economic signal in the forecast model."
        model_body = "Trained on 2010–2026 data across 208 state-category series. 97% accuracy in predicting the direction of inflation movement."
        disclaimer_body = """
        - This tool uses **state-level average data** from DOSM. Actual prices vary by neighbourhood, shop, and lifestyle — this is a planning guide, not a guarantee.
        - Multi-month forecasts use **recursive prediction** — each month's forecast is partly built on the previous month's forecast, so accuracy naturally decreases the further out you look. Treat 12-month forecasts as directional, not precise.
        - Destination spending in the Relocation Planner is **auto-scaled using today's price ratio** between states — it assumes you'll keep the same spending habits, which may not hold if your lifestyle changes significantly.
        - Income and expenditure figures come from the **HIES 2022 survey** and may not reflect current conditions precisely.
        - The official inflation figure shown for comparison is pulled **live from official DOSM national CPI data**, comparing the most recent published month against the same month one year earlier.
        - This is an **independent research project** — not affiliated with DOSM, BNM, or any government agency.
        """
    else:
        hero_title = "Maklumat Inflasi Peribadi Malaysia"
        hero_sub = "Kerajaan menerbitkan satu angka inflasi untuk seluruh negara. Namun, inflasi anda bergantung kepada cara sebenar anda berbelanja. Alat ini mengira kadar inflasi peribadi anda."
        section1_title = "💡 Apakah itu inflasi peribadi?"
        section1_body = (
            "CPI rasmi Malaysia dikira menggunakan 'bakul' tetap kebangsaan — "
            "anggaran purata bagaimana rakyat Malaysia berbelanja merentasi 13 kategori utama.\n\n"
            "Namun, jika anda membelanjakan 40% daripada pendapatan untuk makanan dan harga makanan meningkat sebanyak 5%, "
            "anda akan merasai kesan tersebut dengan lebih ketara berbanding individu yang hanya membelanjakan 20% untuk makanan.\n\n"
            "Inflasi peribadi anda = corak perbelanjaan anda × perubahan harga sebenar setiap kategori."
        )
        example_label = "Contoh"
        example_body = "CPI utama kerajaan: 2.1%<br><br>Individu A (Perbelanjaan makanan tinggi): 2.8% 📈<br>Individu B (Perbelanjaan perumahan tinggi): 1.7% 📉<br><br>Bulan yang sama. Negara yang sama. Realiti yang berbeza."
        section2_title = "🛠️ Fungsi yang ditawarkan oleh aplikasi ini"
        card1_title = "Unjuran Belanjawan"
        card1_body = "Masukkan gaji dan jumlah perbelanjaan anda mengikut kategori dalam unit Ringgit. Kami akan membuat unjuran perbelanjaan anda untuk tempoh 3, 6, atau 12 bulan akan datang, membandingkannya dengan inflasi rasmi, dan mengenal pasti kategori yang paling banyak menelan belanjawan anda secara diam-diam."
        card1_best = "Sesuai untuk: Mengetahui kuasa beli sebenar wang anda pada tahun hadapan."
        card2_title = "Perancang Perpindahan"
        card2_body = "Sedang mempertimbangkan untuk berpindah ke Kuala Lumpur, Johor, atau negeri lain? Kami akan melaraskan perbelanjaan semasa anda secara automatik berdasarkan perbezaan harga sebenar antara negeri, memaparkan unjuran perbandingan dalam unit Ringgit, dan menganggarkan gaji yang diperlukan untuk mengekalkan taraf hidup anda."
        card2_best = "Sesuai untuk: Menilai tawaran pekerjaan di negeri lain."
        cta_text = "👈 Pilih mod daripada bar sisi untuk bermula."
        ready_text = "Sedia untuk mengetahui kadar inflasi peribadi anda?"
        step1_title = "Pilih mod anda"
        step1_body = "Pilih Kalkulator Peribadi atau Perancang Perpindahan daripada bar sisi di sebelah kiri."
        step2_title = "Masukkan angka sebenar anda"
        step2_body = "Masukkan gaji bulanan dan perbelanjaan sebenar anda mengikut kategori dalam unit Ringgit — tiada peratusan atau anggaran diperlukan. Pilih tempoh unjuran yang anda inginkan: 3, 6, atau 12 bulan."
        step3_title = "Semak keputusan anda"
        step3_body = "Lihat perbandingan perbelanjaan anda dengan inflasi rasmi, kenal pasti kategori ketirisan terbesar anda, dan ketahui unjuran lebihan bulanan anda untuk beberapa bulan akan datang."
        howto_header = "📖 Cara menggunakan aplikasi ini — 3 langkah"
        data_header = "📡 Dari manakah sumber data ini diperoleh?"
        data_intro = "Semua data diperoleh daripada API rasmi kerajaan Malaysia — tiada data pihak ketiga atau data anggaran digunakan. Dikemas kini pada setiap bulan."
        disclaimer_header = "⚠️ Penafian penting — sila baca"
        dosm_body = "Jabatan Perangkaan Malaysia (DOSM) — data CPI bulanan mengikut negeri dan kategori, Indeks Harga Pengeluar (PPI), dan penyiasatan pendapatan isi rumah."
        bnm_body = "Data kadar faedah bulanan termasuk Kadar Dasar Semalaman (OPR) — digunakan sebagai isyarat ekonomi dalam model unjuran."
        model_body = "Dilatih menggunakan data dari tahun 2010–2026 merentasi 208 siri negeri-kategori. Mempunyai ketepatan 97% dalam meramal arah pergerakan inflasi."
        disclaimer_body = """
        - Alat ini menggunakan **data purata peringkat negeri** daripada DOSM. Harga sebenar berbeza mengikut kawasan kejiranan, kedai, dan gaya hidup — ini hanyalah panduan perancangan, bukan jaminan mutlak.
        - Unjuran berbilang bulan menggunakan **kaedah ramalan rekursif** — unjuran bagi setiap bulan sebahagiannya terbina daripada unjuran bulan sebelumnya, justeru ketepatannya secara semula jadi akan menurun apabila tempoh unjuran semakin jauh. Jadikan unjuran 12 bulan sebagai panduan asas (indikatif), bukan nilai yang tepat.
        - Perbelanjaan destinasi dalam Perancang Perpindahan **dilaraskan secara automatik menggunakan nisbah harga semasa** antara negeri — fungsi ini mengandaikan anda mengekalkan corak perbelanjaan yang sama, yang mungkin tidak tepat sekiranya gaya hidup anda berubah secara ketara.
        - Angka pendapatan dan perbelanjaan diambil daripada **Penyiasatan HIES 2022** (Penyiasatan Pendapatan dan Perbelanjaan Isi Rumah) dan mungkin tidak mencerminkan keadaan semasa dengan tepat.
        "- Angka inflasi rasmi yang dipaparkan untuk tujuan perbandingan diperoleh **secara lansung daripada data CPI kebangsaan rasmi DOSM**, dengan membandingkan data bulan terkini yang diterbitkan dengan bulan yang sama pada tahun sebelumnya."
        - Ini merupakan **projek penyelidikan bebas** — dan tidak mempunyai sebarang kaitan dengan DOSM, BNM, atau mana-mana agensi kerajaan.
        """

    # ── hero section ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='text-align:center; padding: 40px 20px 20px 20px;'>
        <div style='font-size:3rem;'>🇲🇾</div>
        <h1 style='color:#1F4E79; font-size:2.2rem; margin-bottom:8px;'>
            {hero_title}
        </h1>
        <p style='color:#64748b; font-size:1.1rem; max-width:620px; margin:0 auto;'>
            {hero_sub}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── what is personal inflation ─────────────────────────────────────────────
    st.subheader(section1_title)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        {section1_body}
        """)

    with col2:
        st.markdown(f"""
        <div style='background:#EFF6FF; border-left:4px solid #1F4E79;
                    border-radius:8px; padding:16px 20px;'>
            <b style='color:#1F4E79'>{example_label}</b><br><br>
            {example_body}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── what you can do ────────────────────────────────────────────────────────
    st.subheader(section2_title)

    st.markdown(f"""
     <div style='display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:stretch;'>
        <div style='background:#F8FAFC; border:1px solid #E2E8F0;
                    border-radius:12px; padding:24px; height:100%;
                    box-sizing:border-box;'>
            <div style='font-size:2rem'>📊</div>
            <h3 style='color:#1F4E79; margin:8px 0;'>{card1_title}</h3>
            <p style='color:#475569;'>
                {card1_body}
            </p>
            <p style='color:#1F4E79; font-weight:600'>
                {card1_best}
            </p>
        </div>
        <div style='background:#F8FAFC; border:1px solid #E2E8F0;
                    border-radius:12px; padding:24px; height:100%;
                    box-sizing:border-box;'>
            <div style='font-size:2rem'>🗺️</div>
            <h3 style='color:#1F4E79; margin:8px 0;'>{card2_title}</h3>
            <p style='color:#475569;'>
                {card2_body}
            </p>
            <p style='color:#1F4E79; font-weight:600'>
                {card2_best}
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── how to use ────────────────────────────────────────────────────────────
    st.subheader(howto_header)

    st.markdown(f"""
    <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; align-items:stretch;'>
        <div style='text-align:center; padding:16px; height:100%; box-sizing:border-box;'>
            <div style='font-size:2.5rem'>1️⃣</div>
            <h4 style='color:#1F4E79'>{step1_title}</h4>
            <p style='color:#475569; font-size:0.9rem'>{step1_body}</p>
        </div>
        <div style='text-align:center; padding:16px; height:100%; box-sizing:border-box;'>
            <div style='font-size:2.5rem'>2️⃣</div>
            <h4 style='color:#1F4E79'>{step2_title}</h4>
            <p style='color:#475569; font-size:0.9rem'>{step2_body}</p>
        </div>
        <div style='text-align:center; padding:16px; height:100%; box-sizing:border-box;'>
            <div style='font-size:2.5rem'>3️⃣</div>
            <h4 style='color:#1F4E79'>{step3_title}</h4>
            <p style='color:#475569; font-size:0.9rem'>{step3_body}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── data and model ────────────────────────────────────────────────────────
    st.subheader(data_header)

    st.markdown(f"""{data_intro}""")

    st.markdown(f"""
    <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; align-items:stretch;'>
        <div style='background:#ECFDF5; padding:16px; border-radius:8px;
                    border:1px solid #D1FAE5; height:100%; box-sizing:border-box;'>
            <b style='color:#065F46'>📊 DOSM</b><br>
            <span style='color:#475569; font-size:0.9rem'>{dosm_body}</span>
        </div>
        <div style='background:#EFF6FF; padding:16px; border-radius:8px;
                    border:1px solid #BFDBFE; height:100%; box-sizing:border-box;'>
            <b style='color:#1E40AF'>🏦 Bank Negara Malaysia</b><br>
            <span style='color:#475569; font-size:0.9rem'>{bnm_body}</span>
        </div>
        <div style='background:#FFF7ED; padding:16px; border-radius:8px;
                    border:1px solid #FED7AA; height:100%; box-sizing:border-box;'>
            <b style='color:#92400E'>🤖 {"XGBoost ML Model" if lang=="en" else "Model ML XGBoost"}</b><br>
            <span style='color:#475569; font-size:0.9rem'>{model_body}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── disclaimer ─────────────────────────────────────────────────────────────
    with st.expander(disclaimer_header):
        st.markdown(disclaimer_body)

    st.divider()

    # ── CTA ───────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='text-align:center; padding:20px;'>
        <p style='color:#475569; font-size:1rem; margin-bottom:16px;'>
            {ready_text}
        </p>
        <p style='color:#1F4E79; font-weight:600; font-size:1.1rem;'>
            {cta_text}
        </p>
    </div>
    """, unsafe_allow_html=True)

def calculate_budget_forecast(
    state: str,
    spending: dict,
    monthly_salary: int,
    horizon_months: int,
    lang: str = "en",
) -> dict:
    """
    Orchestrates the full budget forecast:
    - Calls /forecast for each category the user has spending in
    - Computes the household-level summary: felt vs headline inflation,
      biggest leak/saving category, projected surplus
    """
    category_labels = CATEGORIES if lang == "en" else CATEGORY_NAMES_MS
    active_categories = {k: v for k, v in spending.items() if v > 0}

    if not active_categories:
        return None

    category_forecasts = {}
    total_current = 0
    total_projected_final_month = 0
    latest_real_date = None

    for div, amount in active_categories.items():
        result = fetch_forecast(state, div, horizon_months)
        if result is None:
            continue
        
        if latest_real_date is None:
            latest_real_date = result.get("latest_real_date")

        forecast_list = result["forecast"]
        first_index = forecast_list[0]["predicted_index"]
        final_index = forecast_list[-1]["predicted_index"]

        # Scale index change onto the user's actual RM amount for this category
        projected_amounts = [
            amount * (f["predicted_index"] / first_index) for f in forecast_list
        ]

        category_forecasts[div] = {
            "category_name": category_labels.get(div, div),
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
        "latest_real_date": latest_real_date,
    }

# ── personal calculator page ───────────────────────────────────────

def render_personal_calculator(
    state: str,
    spending: dict,
    monthly_salary: int,
    horizon_months: int,
    t: dict,
    lang: str,
):
    st.title(t["budget_forecast_title"])

    total_spent = sum(spending.values())

    if total_spent == 0:
        info_text = "Enter your spending per category in the sidebar to get started." if lang == "en" else "Masukkan perbelanjaan setiap kategori di bar sisi untuk bermula."
        st.info(info_text)
        st.stop()

    if monthly_salary > 0 and total_spent > monthly_salary:
        warn_text = (
            "Your spending exceeds your salary — the forecast will still run, but check your numbers in the sidebar."
            if lang == "en" else
            "Perbelanjaan anda melebihi gaji — ramalan tetap akan dijalankan, tetapi sila semak semula jumlah di bar sisi."
        )
        st.warning(warn_text)

    generate_label = "🚀 Generate My Forecast" if lang == "en" else "🚀 Jana Ramalan Saya"
    generate_hint = (
        "Adjust your salary and spending in the sidebar, then click above to generate your personalised report."
        if lang == "en" else
        "Laraskan gaji dan perbelanjaan anda di bar sisi, kemudian klik butang di atas untuk menjana laporan peribadi anda."
    )
    changed_hint = (
        "⚠️ You've changed your inputs — the report below is based on your previous numbers. Click Generate again to refresh it."
        if lang == "en" else
        "⚠️ Anda telah menukar input — laporan di bawah adalah berdasarkan angka sebelumnya. Klik Jana semula untuk mengemas kininya."
    )

    current_inputs_snapshot = (state, monthly_salary, horizon_months, tuple(sorted(spending.items())))

    if "calculator_generated" not in st.session_state:
        st.session_state.calculator_generated = False
    if "calculator_last_inputs" not in st.session_state:
        st.session_state.calculator_last_inputs = None
    if "calculator_result" not in st.session_state:
        st.session_state.calculator_result = None

    inputs_changed = (
        st.session_state.calculator_generated
        and st.session_state.calculator_last_inputs is not None
        and st.session_state.calculator_last_inputs != current_inputs_snapshot
    )

    col_left, col_btn, col_right = st.columns([1, 2, 1])
    with col_btn:
        if st.button(generate_label, type="primary", use_container_width=True):
            st.session_state.calculator_generated = True
        st.session_state.calculator_last_inputs = current_inputs_snapshot
        # Force a fresh calculation on this exact click
        with st.spinner(f"Forecasting {horizon_months} months ahead..." if lang == "en" else f"Meramal {horizon_months} bulan akan datang..."):
            st.session_state.calculator_result = calculate_budget_forecast(
                state, spending, monthly_salary, horizon_months, lang
            )
        inputs_changed = False  # freshly regenerated, so no longer stale

    if not st.session_state.calculator_generated:
        st.info(generate_hint)
        st.stop()

    if inputs_changed:
        st.warning(changed_hint)

    result = st.session_state.calculator_result

    if result is None:
        st.error("Could not generate a forecast. Please try again." if lang == "en" else "Ramalan tidak dapat dijana. Sila cuba lagi.")
        st.stop()

    latest_date_display = result.get("latest_real_date", "—")

    caption_text = (
        f"Based on your actual spending in {state} · {horizon_months}-month forecast · Latest data: {latest_date_display}"
        if lang == "en" else
        f"Berdasarkan perbelanjaan sebenar anda di {state} · Ramalan {horizon_months} bulan · Data terkini: {latest_date_display}"
    )
    st.caption(caption_text)

    if inputs_changed:
        st.warning(changed_hint)

    # ── Report header — gives the output a clear name and identity ─────────
    report_title = "📄 Your Personal Inflation Report" if lang == "en" else "📄 Laporan Inflasi Peribadi Anda"
    st.subheader(report_title)

    # ── Plain-language interpretation — tells the user what the number means ─
    felt = result["felt_inflation_pct"]

    headline_data = fetch_headline_inflation()
    headline_pct = headline_data["headline_inflation_pct"] if headline_data else 1.5
    gap = abs(felt - headline_pct)

    if gap < 0.5:
        interpretation = (
            "Your personal inflation is close to the national average — no urgent action needed, but it's worth keeping an eye on the category flagged below."
            if lang == "en" else
            "Inflasi peribadi anda hampir sama dengan purata kebangsaan — tiada tindakan segera diperlukan, tetapi elok pantau kategori yang dikenal pasti di bawah."
        )
        interp_color = "#0F6E5C"
    elif felt > headline_pct:
        interpretation = (
            "Your personal inflation is running noticeably higher than the national average — the category flagged as your biggest leak below is worth addressing first."
            if lang == "en" else
            "Inflasi peribadi anda jauh lebih tinggi berbanding purata kebangsaan — kategori yang dikenal pasti sebagai paling banyak makan bajet di bawah wajar ditangani dahulu."
        )
        interp_color = "#dc2626"
    else:
        interpretation = (
            "Your personal inflation is running lower than the national average — your spending pattern is currently working in your favour."
            if lang == "en" else
            "Inflasi peribadi anda lebih rendah berbanding purata kebangsaan — corak perbelanjaan anda pada masa ini memberi manfaat kepada anda."
        )
        interp_color = "#16a34a"

    st.markdown(f"""
    <p style='color:{interp_color}; font-size:1.05rem; font-weight:500; margin-bottom:16px;'>
        {interpretation}
    </p>
    """, unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t["current_spending_label"], f"RM{result['total_current']:,.0f}")
    with col2:
        delta_rm = result["total_projected_final"] - result["total_current"]
        st.metric(
            t["projected_spending_label"].format(n=horizon_months),
            f"RM{result['total_projected_final']:,.0f}",
            delta=f"{delta_rm:+,.0f} RM",
            delta_color="inverse",
        )
    with col3:
        surplus_delta = result["projected_surplus"] - result["current_surplus"]
        st.metric(
            t["projected_surplus_label"],
            f"RM{result['projected_surplus']:,.0f}",
            delta=f"{surplus_delta:+,.0f} RM",
            delta_color="normal",
        )

    st.divider()

    leak = result["biggest_leak"]
    rising_word = "Rising" if lang == "en" else "Meningkat"
    watch_text = "watch this category closely." if lang == "en" else "kategori ini wajar dipantau dengan teliti."

    save = result["biggest_saving"]
    if lang == "en":
        save_word = "falling" if save["change_pct"] < 0 else "rising slowest"
        exposed_text = "this is where you're least exposed."
    else:
        save_word = "menurun" if save["change_pct"] < 0 else "Meningkat pada kadar paling perlahan"
        exposed_text = "kategori ini memberikan kesan paling minimum kepada anda."

    st.markdown(f"""
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:stretch;'>
        <div style='background:#FEF2F2; border-left:4px solid #dc2626;
                    border-radius:8px; padding:16px 20px; height:100%;
                    box-sizing:border-box;'>
            <b style='color:#991B1B'>{t['biggest_leak_label']}</b><br><br>
            <b>{leak['name']}</b><br>
            <span style='color:#7F1D1D'>{rising_word} {leak['change_pct']:+.1f}% {"over" if lang=="en" else "dalam"} {horizon_months} {"months" if lang=="en" else "bulan"} — {watch_text}</span>
        </div>
        <div style='background:#F0FDF4; border-left:4px solid #16a34a;
                    border-radius:8px; padding:16px 20px; height:100%;
                    box-sizing:border-box;'>
            <b style='color:#166534'>{t['best_category_label']}</b><br><br>
            <b>{save['name']}</b><br>
            <span style='color:#14532D'>{save_word.capitalize()} ({save['change_pct']:+.1f}%) — {exposed_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    
    # ── prescriptive insight — auto-generated suggestion ────────────────────
    insight_data = fetch_prescriptive_insight("personal_calculator", {
        "biggest_leak_name": result["biggest_leak"]["name"],
        "biggest_leak_pct": result["biggest_leak"]["change_pct"],
        "biggest_saving_name": result["biggest_saving"]["name"],
        "current_surplus": result["current_surplus"],
        "projected_surplus": result["projected_surplus"],
        "horizon_months": horizon_months,
    })

    if insight_data and insight_data.get("insight"):
        insight_label = "💡 What you could consider" if lang == "en" else "💡 Apa yang boleh anda pertimbangkan"
        ai_disclaimer = (
            "This suggestion is generated by AI (Google Gemini) based on your numbers — it is general guidance, not financial advice."
            if lang == "en" else
            "Cadangan ini dijana oleh AI (Google Gemini) berdasarkan angka anda — ia adalah panduan umum, bukan nasihat kewangan."
        )
        st.markdown(f"""
        <div style='background:#F0F9FF; border:1px solid #BAE6FD; border-radius:10px;
                    padding:16px 20px; margin:16px 0;'>
            <b style='color:#0369A1'>{insight_label}</b><br><br>
            <span style='color:#0C4A6E'>{insight_data["insight"]}</span>
            <hr style='border:none; border-top:1px solid #BAE6FD; margin:12px 0 8px 0;'>
            <span style='font-size:0.78rem; color:#64748B; font-style:italic;'>🤖 {ai_disclaimer}</span>
        </div>
        """, unsafe_allow_html=True)

    direction_word = ("higher" if felt > headline_pct else "lower") if lang == "en" else ("lebih tinggi" if felt > headline_pct else "lebih rendah")
    validation_color = "#dc2626" if felt > headline_pct else "#16a34a"

    if lang == "en":
        reality_text = (
            f"Official inflation is currently around <b>{headline_pct}%</b>. "
            f"Based on <b>your actual spending</b>, your personal inflation rate over the next {horizon_months} months "
            f"is projected at <b style='color:{validation_color}'>{felt:+.2f}%</b> — "
            f"{abs(felt - headline_pct):.1f} points {direction_word} than the official number."
        )
    else:
        reality_text = (
            f"Kadar inflasi rasmi pada masa ini adalah sekitar <b>{headline_pct}%</b>. "
            f"Berdasarkan <b>perbelanjaan sebenar anda</b>, kadar inflasi peribadi anda bagi tempoh {horizon_months} bulan akan datang diunjurkan pada <b style='color:{validation_color}'>{felt:+.2f}%</b> — "
            f"iaitu {abs(felt - headline_pct):.1f} mata {direction_word} berbanding kadar rasmi."
        )

        st.markdown(f"""
    <div class="summary-box" style="border-left-color:{validation_color}">
        <strong>{t['reality_check_label']}</strong><br><br>
        {reality_text}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader(t["category_forecasts_header"])
    caption2 = (
        "How much you'll spend per category, with a widening confidence range the further out we forecast."
        if lang == "en" else
        "Jumlah perbelanjaan anda mengikut kategori, dengan julat keyakinan yang semakin melebar apabila tempoh unjuran semakin jauh."
    )
    st.caption(caption2)

    sorted_cats = sorted(result["category_forecasts"].items(), key=lambda x: x[1]["current_rm"], reverse=True)

    for div, cat_data in sorted_cats[:4]:
        fig = render_forecast_chart(cat_data["forecast_raw"], cat_data["category_name"], base_amount_rm=cat_data["current_rm"])
        st.plotly_chart(fig, use_container_width=True)

    if len(sorted_cats) > 4:
        more_text = f"Show {len(sorted_cats) - 4} more categories" if lang == "en" else f"Papar {len(sorted_cats) - 4} kategori lagi"
        with st.expander(more_text):
            for div, cat_data in sorted_cats[4:]:
                fig = render_forecast_chart(cat_data["forecast_raw"], cat_data["category_name"], base_amount_rm=cat_data["current_rm"])
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader(t["full_breakdown_header"])

    if lang == "en":
        cols = {"Category": "Category", "Current": "Current (RM)", "Projected": f"Projected month {horizon_months} (RM)", "Change": "Change"}
    else:
        cols = {"Category": "Kategori", "Current": "Semasa (RM)", "Projected": f"Anggaran bulan {horizon_months} (RM)", "Change": "Perubahan"}

    table_data = []
    for div, cat_data in sorted_cats:
        table_data.append({
            cols["Category"]: cat_data["category_name"],
            cols["Current"]: f"RM{cat_data['current_rm']:,.0f}",
            cols["Projected"]: f"RM{cat_data['projected_final_rm']:,.0f}",
            cols["Change"]: f"{cat_data['change_rm']:+,.0f} RM ({cat_data['change_pct']:+.1f}%)",
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    footer_note = (
        "⚠️ Forecasts use recursive prediction — accuracy naturally decreases further into the horizon as the model builds on its own prior predictions. Treat the widening range as a guide, not a guarantee."
        if lang == "en" else
        "⚠️ Unjuran ini menggunakan kaedah ramalan rekursif — ketepatan secara semula jadi akan menurun apabila tempoh unjuran semakin jauh memandangkan model ini terbina berdasarkan ramalannya yang terdahulu. Jadikan julat yang semakin melebar ini sebagai panduan, bukan satu jaminan mutlak."
    )
    st.caption(footer_note)

    st.divider()

    # ── Ask About This App — chat interface, visually distinct ─────────────
    st.markdown(f"""
    <div style='background:#FDF4FF; border:1px solid #F0ABFC; border-radius:12px;
                padding:20px 24px; margin:16px 0 8px 0;'>
        <h3 style='color:#86198F; margin:0 0 4px 0;'>
            💬 {"Ask Anything" if lang == "en" else "Tanya Apa Sahaja"}
        </h3>
        <p style='color:#701A75; font-size:0.9rem; margin:0;'>
            {"Ask about your forecast, or how any part of this app works — powered by AI, grounded in your real data." if lang == "en" else "Tanya tentang unjuran anda, atau bagaimana mana-mana bahagian aplikasi ini berfungsi — dikuasakan AI, berdasarkan data sebenar anda."}
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "ask_chat_history" not in st.session_state:
        st.session_state.ask_chat_history = []

    for msg in st.session_state.ask_chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    ask_placeholder = (
        "Ask a question about your forecast or this app..."
        if lang == "en" else
        "Tanya soalan tentang unjuran anda atau aplikasi ini..."
    )
    question = st.chat_input(ask_placeholder, key="ask_chat_input_pc")

    if question:
        st.session_state.ask_chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Default to the biggest-leak category if the question seems personal —
        # no pre-selector needed, this is just a sensible fallback
        default_div = next(
            (d for d, cat_data in result["category_forecasts"].items()
             if cat_data["category_name"] == result["biggest_leak"]["name"]),
            None
        )

        spinner_text_ask = "Thinking..." if lang == "en" else "Sedang berfikir..."
        with st.chat_message("assistant"):
            with st.spinner(spinner_text_ask):
                ask_result = fetch_ask_answer(question, state, default_div, horizon_months)

            if ask_result.get("error"):
                error_msg = "Could not get an answer right now." if lang == "en" else "Tidak dapat menjawab sekarang."
                st.error(error_msg)
                st.session_state.ask_chat_history.append({"role": "assistant", "content": error_msg})
            else:
                st.write(ask_result["answer"])
                ai_note = "🤖 AI-generated, grounded in retrieved app knowledge and your data where relevant." if lang == "en" else "🤖 Dijana AI, berdasarkan pengetahuan aplikasi dan data anda jika relevan."
                st.caption(ai_note)
                st.session_state.ask_chat_history.append({"role": "assistant", "content": ask_result["answer"]})

# ── relocation planner page ───────────────────────────────────────────────────

def render_relocation_planner(
    origin_state: str,
    spending: dict,
    monthly_salary: int,
    t: dict,
    lang: str,
):
    """
    Relocation Planner — RM-based comparison between two states.

    Auto-suggests destination spending by scaling origin RM amounts
    by the current price ratio between states, then lets the user
    adjust before running a full multi-month forecast comparison.
    """
    if lang == "en":
        page_caption = "Compare your real cost of living between two Malaysian states, in ringgit — using official DOSM and BNM data"
        moving_from_label = "Moving from:"
        moving_to_label = "Moving to"
        suggested_header = "Suggested spending in your new state"
        suggested_caption = "Auto-scaled from your {origin} spending using current {dest} prices. Adjust any amount if you expect your habits to change."
        editable_label = "Your destination budget (editable):"
        compare_button_text = "🔍 Compare full forecast"
        compare_hint = "Click above to run the full multi-month comparison."
        warning_zero = "Set at least one destination spending amount above."
        spinner_text = "Forecasting {n} months for both states..."
        error_text = "Could not generate forecasts. Please try again."
        summary_title = "📊 Relocation Summary"
        salary_needed_label = "Recommended minimum salary in {state}:"
        category_compare_header = "Category-by-category comparison"
        show_more_text = "Show {n} more categories"
        disclaimer_text = "⚠️ Destination amounts are auto-suggested by scaling your current spending using today's price ratio between states, then forecast forward independently for each state. Forecasts use recursive prediction — treat the further-out months as directional, not exact."
    else:
        page_caption = "Bandingkan kos sara hidup sebenar anda antara dua negeri di Malaysia, dalam unit Ringgit — menggunakan data rasmi DOSM dan BNM"
        moving_from_label = "Berpindah dari:"
        moving_to_label = "Berpindah ke"
        suggested_header = "Cadangan perbelanjaan di negeri baharu anda"
        suggested_caption = "Dilaraskan secara automatik daripada perbelanjaan {origin} anda menggunakan harga semasa di {dest}. Ubah mana-mana jumlah jika anda menjangkakan perubahan pada corak perbelanjaan anda."
        editable_label = "Belanjawan destinasi anda (boleh disunting):"
        compare_button_text = "🔍 Bandingkan unjuran penuh"
        compare_hint = "Klik butang di atas untuk menjana perbandingan penuh bagi beberapa bulan."
        warning_zero = "Sila tetapkan sekurang-kurangnya satu jumlah perbelanjaan destinasi di atas."
        spinner_text = "Menjana unjuran {n} bulan untuk kedua-dua negeri..."
        error_text = "Unjuran tidak dapat dijana. Sila cuba lagi."
        summary_title = "📊 Ringkasan Perpindahan"
        salary_needed_label = "Gaji minimum yang disyorkan di {state}:"
        category_compare_header = "Perbandingan mengikut kategori"
        show_more_text = "Papar {n} kategori lagi"
        disclaimer_text = "⚠️ Jumlah destinasi dicadangkan secara automatik dengan melaraskan perbelanjaan semasa anda menggunakan nisbah harga hari ini antara negeri, kemudian diunjurkan ke hadapan secara berasingan bagi setiap negeri. Unjuran ini menggunakan kaedah ramalan rekursif — jadikan unjuran bulan-bulan yang lebih jauh sebagai panduan asas (indikatif), bukan nilai tepat."
    
    category_labels = CATEGORIES if lang == "en" else CATEGORY_NAMES_MS
    st.title(t["relocation_title"])
    st.caption(page_caption)

    active_spending = {k: v for k, v in spending.items() if v > 0}

    if not active_spending:
        st.info("Enter your spending in the sidebar to get started.")
        st.stop()

    # ── destination state + horizon selectors ───────────────────────────────
    col_origin, col_arrow, col_dest = st.columns([2, 0.6, 2])

    with col_origin:
        st.markdown(f"""
        <div style='background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px;
                    padding:10px 16px; height:58px; display:flex; flex-direction:column;
                    justify-content:center; box-sizing:border-box;'>
            <span style='font-size:0.8rem; color:#1E40AF; font-weight:600'>{moving_from_label}</span>
            <span style='font-size:1.05rem; color:#1A1A1A;'>{origin_state}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_arrow:
        st.markdown(
            "<div style='text-align:center;font-size:1.6rem;height:58px;"
            "display:flex;align-items:center;justify-content:center;color:#94A3B8;'>→</div>",
            unsafe_allow_html=True,
        )

    with col_dest:
        destination_state = st.selectbox(
            moving_to_label,
            options=[s for s in STATES if s != origin_state],
            index=STATES.index("W.P. Kuala Lumpur")
            if origin_state != "W.P. Kuala Lumpur"
            else 0,
            label_visibility="collapsed",
        )

    horizon_months = st.radio(
        t["horizon_label"],
        options=[3, 6, 12],
        index=1,
        format_func=lambda x: f"{x} months" if lang == "en" else f"{x} bulan",
        horizontal=True,
        key="reloc_horizon",
    )

    st.divider()

# ── fetch price ratios for every active category ────────────────────────
    st.subheader(suggested_header)
    st.caption(suggested_caption.format(origin=origin_state, dest=destination_state))

    fetch_ratios_label = "📊 Get Suggested Spending" if lang == "en" else "📊 Dapatkan Cadangan Perbelanjaan"
    fetch_ratios_hint = (
        "Confirm your destination state above, then click here to see auto-scaled spending suggestions."
        if lang == "en" else
        "Sahkan negeri destinasi anda di atas, kemudian klik di sini untuk melihat cadangan perbelanjaan yang dianggarkan secara automatik."
    )
    ratios_changed_hint = (
        "⚠️ You've changed your destination or horizon — the suggestions below are for your previous choice. Click above to refresh."
        if lang == "en" else
        "⚠️ Anda telah menukar destinasi atau tempoh — cadangan di bawah adalah untuk pilihan sebelumnya. Klik di atas untuk kemas kini."
    )

    current_ratio_snapshot = (origin_state, destination_state, tuple(sorted(active_spending.items())))

    if "reloc_ratios_generated" not in st.session_state:
        st.session_state.reloc_ratios_generated = False
    if "reloc_ratios_last_inputs" not in st.session_state:
        st.session_state.reloc_ratios_last_inputs = None
    if "reloc_destination_spending" not in st.session_state:
        st.session_state.reloc_destination_spending = None
    if "reloc_ratio_info" not in st.session_state:
        st.session_state.reloc_ratio_info = None

    ratios_changed = (
        st.session_state.reloc_ratios_generated
        and st.session_state.reloc_ratios_last_inputs is not None
        and st.session_state.reloc_ratios_last_inputs != current_ratio_snapshot
    )

    col_left_r1, col_btn_r1, col_right_r1 = st.columns([1, 2, 1])
    with col_btn_r1:
        if st.button(fetch_ratios_label, type="primary", use_container_width=True):
            st.session_state.reloc_ratios_generated = True
        st.session_state.reloc_ratios_last_inputs = current_ratio_snapshot
        ratios_changed = False

        destination_spending = {}
        ratio_info = {}

        with st.spinner(f"Comparing prices between {origin_state} and {destination_state}..."):
            for div, origin_amount in active_spending.items():
                ratio_data = fetch_price_ratio(origin_state, destination_state, div)

                if ratio_data:
                    ratio = ratio_data["ratio"]
                    pct_diff = ratio_data["pct_difference"]
                else:
                    ratio = 1.0
                    pct_diff = 0.0

                suggested_amount = round(origin_amount * ratio)
                ratio_info[div] = {"ratio": ratio, "pct_diff": pct_diff}
                destination_spending[div] = suggested_amount

        st.session_state.reloc_destination_spending = destination_spending
        st.session_state.reloc_ratio_info = ratio_info

    if not st.session_state.reloc_ratios_generated:
        st.info(fetch_ratios_hint)
        st.stop()

    if ratios_changed:
        st.warning(ratios_changed_hint)

    destination_spending = st.session_state.reloc_destination_spending
    ratio_info = st.session_state.reloc_ratio_info

    # ── editable destination table ───────────────────────────────────────────
    st.markdown(f"**{editable_label}**")

    edited_spending = {}
    for div, suggested in destination_spending.items():
        col_name, col_origin_amt, col_dest_amt, col_diff = st.columns([2.5, 1.3, 1.5, 1.2])

        with col_name:
            st.markdown(f"{category_labels.get(div, div)}")
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
        st.warning(warning_zero)
        st.stop()

    changed_hint_reloc = (
        "⚠️ You've changed your inputs — the comparison below is based on your previous numbers. Click Compare again to refresh it."
        if lang == "en" else
        "⚠️ Anda telah menukar input — perbandingan di bawah adalah berdasarkan angka sebelumnya. Klik Bandingkan semula untuk mengemas kininya."
    )

    current_reloc_snapshot = (
        origin_state, destination_state, horizon_months,
        tuple(sorted(active_spending.items())),
        tuple(sorted(edited_spending.items())),
    )

    if "reloc_generated" not in st.session_state:
        st.session_state.reloc_generated = False
    if "reloc_last_inputs" not in st.session_state:
        st.session_state.reloc_last_inputs = None
    if "reloc_origin_results" not in st.session_state:
        st.session_state.reloc_origin_results = None
    if "reloc_dest_results" not in st.session_state:
        st.session_state.reloc_dest_results = None

    reloc_inputs_changed = (
        st.session_state.reloc_generated
        and st.session_state.reloc_last_inputs is not None
        and st.session_state.reloc_last_inputs != current_reloc_snapshot
    )

    col_left_r2, col_btn_r2, col_right_r2 = st.columns([1, 2, 1])
    with col_btn_r2:
        if st.button(compare_button_text, type="primary", use_container_width=True):
            st.session_state.reloc_generated = True
        st.session_state.reloc_last_inputs = current_reloc_snapshot
        reloc_inputs_changed = False

        origin_results = {}
        dest_results = {}

        with st.spinner(spinner_text.format(n=horizon_months)):
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

        st.session_state.reloc_origin_results = origin_results
        st.session_state.reloc_dest_results = dest_results

    if not st.session_state.reloc_generated:
        st.caption(compare_hint)
        st.stop()

    if reloc_inputs_changed:
        st.warning(changed_hint_reloc)

    origin_results = st.session_state.reloc_origin_results
    dest_results = st.session_state.reloc_dest_results

    if not origin_results or not dest_results:
        st.error(error_text)
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
            "category_name": category_labels.get(div, div),
            "origin_amounts": origin_amounts,
            "dest_amounts": dest_amounts,
            "origin_raw": origin_raw,
            "dest_raw": dest_raw,
        }

    st.divider()

    # ── Ask About This App — chat interface, visually distinct ─────────────
    st.markdown(f"""
    <div style='background:#FDF4FF; border:1px solid #F0ABFC; border-radius:12px;
                padding:20px 24px; margin:16px 0 8px 0;'>
        <h3 style='color:#86198F; margin:0 0 4px 0;'>
            💬 {"Ask Anything" if lang == "en" else "Tanya Apa Sahaja"}
        </h3>
        <p style='color:#701A75; font-size:0.9rem; margin:0;'>
            {"Ask about this comparison, or how any part of this app works — powered by AI, grounded in your real data." if lang == "en" else "Tanya tentang perbandingan ini, atau bagaimana mana-mana bahagian aplikasi ini berfungsi — dikuasakan AI, berdasarkan data sebenar anda."}
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "ask_chat_history_reloc" not in st.session_state:
        st.session_state.ask_chat_history_reloc = []

    for msg in st.session_state.ask_chat_history_reloc:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    ask_placeholder_reloc = (
        "Ask a question about this comparison or the app..."
        if lang == "en" else
        "Tanya soalan tentang perbandingan ini atau aplikasi..."
    )
    question_reloc = st.chat_input(ask_placeholder_reloc, key="ask_chat_input_reloc")

    if question_reloc:
        st.session_state.ask_chat_history_reloc.append({"role": "user", "content": question_reloc})
        with st.chat_message("user"):
            st.write(question_reloc)

        # Default to the category with the biggest cost difference, if relevant
        default_div_reloc = max(
            category_comparisons.items(),
            key=lambda item: abs(item[1]["dest_amounts"][-1] - item[1]["origin_amounts"][-1]),
            default=(None, None),
        )[0]

        spinner_text_ask_reloc = "Thinking..." if lang == "en" else "Sedang berfikir..."
        with st.chat_message("assistant"):
            with st.spinner(spinner_text_ask_reloc):
                ask_result_reloc = fetch_ask_answer(question_reloc, origin_state, default_div_reloc, horizon_months)

            if ask_result_reloc.get("error"):
                error_msg = "Could not get an answer right now." if lang == "en" else "Tidak dapat menjawab sekarang."
                st.error(error_msg)
                st.session_state.ask_chat_history_reloc.append({"role": "assistant", "content": error_msg})
            else:
                st.write(ask_result_reloc["answer"])
                ai_note = "🤖 AI-generated, grounded in retrieved app knowledge and your data where relevant." if lang == "en" else "🤖 Dijana AI, berdasarkan pengetahuan aplikasi dan data anda jika relevan."
                st.caption(ai_note)
                st.session_state.ask_chat_history_reloc.append({"role": "assistant", "content": ask_result_reloc["answer"]})

    # ── summary banner ────────────────────────────────────────────────────────
    cost_diff_rm = dest_total_final - origin_total_final
    cost_diff_pct = (cost_diff_rm / origin_total_final * 100) if origin_total_final else 0
    salary_needed = round(dest_total_final * 1.30)

    direction = "more" if cost_diff_rm > 0 else "less"
    if lang == "en":
        summary_body = (
            f"By month {horizon_months}, your monthly spending in "
            f"<b>{destination_state}</b> is projected at "
            f"<b>RM{dest_total_final:,.0f}</b> — that's "
            f"<b>RM{abs(cost_diff_rm):,.0f} {direction}</b> "
            f"({cost_diff_pct:+.1f}%) than staying in <b>{origin_state}</b> "
            f"(projected RM{origin_total_final:,.0f})."
        )
        buffer_note = "includes a 30% buffer"
    else:
        direction_ms = "lebih tinggi" if cost_diff_rm > 0 else "lebih rendah"
        summary_body = (
            f"Menjelang bulan ke-{horizon_months}, perbelanjaan bulanan anda di "
            f"<b>{destination_state}</b> dianggarkan sebanyak "
            f"<b>RM{dest_total_final:,.0f}</b> — iaitu "
            f"<b>RM{abs(cost_diff_rm):,.0f} {direction_ms}</b> "
            f"({cost_diff_pct:+.1f}%) berbanding kekal di <b>{origin_state}</b> "
            f"(dianggarkan RM{origin_total_final:,.0f})."
        )
        buffer_note = "termasuk margin keselamatan 30%"

    st.markdown(f"""
    <div class="summary-box">
        <strong>{summary_title}</strong><br><br>
        {summary_body}<br><br>
        {salary_needed_label.format(state=destination_state)}
        <b>RM{salary_needed:,.0f}/{"bulan" if lang=="ms" else "month"}</b> ({buffer_note}).
    </div>
    """, unsafe_allow_html=True)

    reloc_insight_data = fetch_prescriptive_insight("relocation_planner", {
        "origin_state": origin_state,
        "destination_state": destination_state,
        "cost_diff_rm": cost_diff_rm,
        "salary_needed": salary_needed,
        "horizon_months": horizon_months,
    })

    if reloc_insight_data and reloc_insight_data.get("insight"):
        insight_label = "💡 What you could consider" if lang == "en" else "💡 Apa yang boleh anda pertimbangkan"
        ai_disclaimer = (
            "This suggestion is generated by AI (Google Gemini) based on your numbers above — it is general guidance, not financial advice."
            if lang == "en" else
            "Cadangan ini dijana oleh AI (Google Gemini) berdasarkan angka anda di atas — ia adalah panduan umum, bukan nasihat kewangan."
        )
        st.markdown(f"""
        <div style='background:#F0F9FF; border:1px solid #BAE6FD; border-radius:10px;
                    padding:16px 20px; margin:16px 0;'>
            <b style='color:#0369A1'>{insight_label}</b><br><br>
            <span style='color:#0C4A6E'>{reloc_insight_data["insight"]}</span>
            <hr style='border:none; border-top:1px solid #BAE6FD; margin:12px 0 8px 0;'>
            <span style='font-size:0.78rem; color:#64748B; font-style:italic;'>🤖 {ai_disclaimer}</span>
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
    st.subheader(category_compare_header)

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
        with st.expander(show_more_text.format(n=len(sorted_comparisons) - 4)):
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
    st.caption(disclaimer_text)

def render_data_explorer(t, lang, state):
    st.title(t["explorer_title"])
    st.caption(t["explorer_caption"])

    # ── National overview — the big picture, before drilling down ───────────
    st.subheader(t["national_overview_header"])
    st.caption(t["national_overview_caption"])

    headline_data = fetch_headline_inflation()
    if headline_data:
        col_headline, col_spacer = st.columns([1, 2])
        with col_headline:
            st.metric(
                "National inflation (YoY)" if lang == "en" else "Inflasi Kebangsaan (YoY)",
                f"{headline_data['headline_inflation_pct']:+.2f}%",
            )
            st.caption(f"As of {headline_data['latest_date'][:7]}" if lang == "en" else f"Setakat {headline_data['latest_date'][:7]}")

    st.markdown(f"**{t['category_leaderboard_header']}**")
    st.caption(t["category_leaderboard_caption"])

    leaderboard_data = fetch_category_leaderboard()
    if leaderboard_data:
        cat_names = [c["category_name"] for c in leaderboard_data["categories"]]
        cat_values = [c["yoy_change_pct"] for c in leaderboard_data["categories"]]
        cat_colors = ["#dc2626" if v > 0 else "#16a34a" for v in cat_values]

        fig_leaderboard = go.Figure(go.Bar(
            x=cat_values, y=cat_names, orientation="h",
            marker_color=cat_colors,
            text=[f"{v:+.1f}%" for v in cat_values],
            textposition="outside",
        ))
        fig_leaderboard.update_layout(
            height=380, margin=dict(t=10, b=30, l=200, r=60),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Year-on-year change (%)" if lang == "en" else "Perubahan tahun ke tahun (%)",
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig_leaderboard, use_container_width=True)

    st.divider()

    category_labels = CATEGORIES if lang == "en" else CATEGORY_NAMES_MS

    st.markdown(f"**{t['state_leaderboard_header']}**")
    st.caption(t["state_leaderboard_caption"])

    state_lb_division = st.selectbox(
        t["explorer_category_label"],
        options=list(category_labels.keys()),
        format_func=lambda k: category_labels.get(k, k),
        key="state_lb_division",
    )

    state_lb_data = fetch_state_leaderboard(state_lb_division)
    if state_lb_data:
        state_names = [s["state"] for s in state_lb_data["states"]]
        state_values = [s["cpi_index"] for s in state_lb_data["states"]]
        highlight_colors = ["#1F4E79" if s == state else "#94A3B8" for s in state_names]

        fig_state_lb = go.Figure(go.Bar(
            x=state_values, y=state_names, orientation="h",
            marker_color=highlight_colors,
            text=[f"{v:.1f}" for v in state_values],
            textposition="outside",
        ))
        fig_state_lb.update_layout(
            height=440, margin=dict(t=10, b=30, l=140, r=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="CPI Index" if lang == "en" else "Indeks CPI",
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig_state_lb, use_container_width=True)
        st.caption(
            f"Your state ({state}) highlighted in blue."
            if lang == "en" else
            f"Negeri anda ({state}) ditonjolkan dalam biru."
        )

    st.divider()
    st.info(t["explore_further_text"])
    st.divider()

    st.caption(
        f"Showing data for **{state}** — change this in the sidebar."
        if lang == "en" else
        f"Menunjukkan data untuk **{state}** — tukar di bar sisi."
    )

    col_cat, col_range = st.columns(2)
    with col_cat:
        explorer_division = st.selectbox(
            t["explorer_category_label"],
            options=list(category_labels.keys()),
            format_func=lambda k: category_labels.get(k, k),
            key="explorer_division",
        )
    with col_range:
        months_back = st.selectbox(
            t["explorer_range_label"],
            options=[24, 36, 60, 120],
            format_func=lambda x: f"{x} months" if lang == "en" else f"{x} bulan",
            index=2,
            key="explorer_months",
        )

    with st.spinner("Loading historical data..." if lang == "en" else "Memuatkan data sejarah..."):
        trends = fetch_trends(state, explorer_division, months_back)

    if trends is None:
        st.error("Could not load historical data." if lang == "en" else "Data sejarah tidak dapat dimuatkan.")
        st.stop()

    series = trends["series"]
    dates = [row["date"] for row in series]

    st.divider()

    # ── CPI trend with 3-month moving average overlay ──────────────────────
    st.subheader(t["cpi_trend_header"])
    st.caption(
        "The 3-month average is the strongest signal our model uses to forecast your spending."
        if lang == "en" else
        "Purata 3 bulan adalah isyarat paling kuat yang digunakan model kami untuk meramal perbelanjaan anda."
    )

    fig_cpi = go.Figure()
    fig_cpi.add_trace(go.Scatter(
        x=dates, y=[row["cpi_index"] for row in series],
        mode="lines", name="CPI Index", line=dict(color="#64748b", width=1.5),
    ))
    fig_cpi.add_trace(go.Scatter(
        x=dates, y=[row["cpi_ma_3"] for row in series],
        mode="lines", name="3-month average", line=dict(color="#1F4E79", width=3),
    ))

     # ── known economic events, marked only if they fall within the visible date range ──
    known_events = [
        {
            "date": "2024-03-01",
            "label_en": "SST raised 6%→8%",
            "label_ms": "SST naik 6%→8%",
        },
        {
            "date": "2024-06-01",
            "label_en": "Diesel subsidy cut",
            "label_ms": "Subsidi diesel dikurangkan",
        },
    ]

    chart_dates = set(dates)
    date_to_cpi = {row["date"]: row["cpi_index"] for row in series}

    event_x, event_y, event_hover = [], [], []
    for event in known_events:
        if event["date"] in chart_dates:
            fig_cpi.add_vline(
                x=event["date"],
                line_width=1.5,
                line_dash="dash",
                line_color="#D4890A",
            )
            event_x.append(event["date"])
            event_y.append(date_to_cpi[event["date"]])
            label = event["label_en"] if lang == "en" else event["label_ms"]
            event_hover.append(f"{event['date']}<br>{label}")

    if event_x:
        fig_cpi.add_trace(go.Scatter(
            x=event_x, y=event_y,
            mode="markers",
            marker=dict(size=10, color="#D4890A", symbol="diamond"),
            name="Policy events" if lang == "en" else "Peristiwa dasar",
            hovertext=event_hover,
            hoverinfo="text",
        ))

    fig_cpi.update_layout(
        height=380, margin=dict(t=20, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cpi, use_container_width=True)
    st.caption(
        "Dashed markers: verified policy events (SST rate increase, 1 March 2024; targeted diesel subsidy, 10 June 2024 — Ministry of Finance Malaysia)."
        if lang == "en" else
        "Penanda putus-putus: peristiwa dasar yang disahkan (kenaikan kadar SST, 1 Mac 2024; subsidi diesel bersasar, 10 Jun 2024 — Kementerian Kewangan Malaysia)."
    )

    st.divider()

    # ── PPI vs CPI ───────────────────────────────────────────────────────────
    st.subheader(t["ppi_overlay_header"])
    st.caption(
        "Producer prices often move before consumer prices — a leading signal our model captures as ppi_lag features."
        if lang == "en" else
        "Harga pengeluar sering bergerak dahulu berbanding harga pengguna — isyarat awal yang ditangkap model kami sebagai ciri ppi_lag."
    )

    fig_ppi = go.Figure()
    fig_ppi.add_trace(go.Scatter(
        x=dates, y=[row["cpi_index"] for row in series],
        mode="lines", name="CPI (consumer)", line=dict(color="#1F4E79", width=2.5),
    ))
    fig_ppi.add_trace(go.Scatter(
        x=dates, y=[row["ppi_index"] for row in series],
        mode="lines", name="PPI (producer)", line=dict(color="#dc2626", width=2, dash="dot"),
    ))
    fig_ppi.update_layout(
        height=380, margin=dict(t=20, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ppi, use_container_width=True)

    st.divider()

    # ── OPR vs CPI (dual y-axis) ────────────────────────────────────────────
    st.subheader(t["opr_overlay_header"])
    st.caption(
        "BNM's interest rate decisions (OPR) are one of the signals our model uses, shown here against inflation trend."
        if lang == "en" else
        "Keputusan kadar faedah BNM (OPR) adalah salah satu isyarat yang digunakan model kami, ditunjukkan di sini berbanding trend inflasi."
    )

    fig_opr = go.Figure()
    fig_opr.add_trace(go.Scatter(
        x=dates, y=[row["cpi_ma_3"] for row in series],
        mode="lines", name="CPI trend", line=dict(color="#1F4E79", width=2.5), yaxis="y1",
    ))
    fig_opr.add_trace(go.Scatter(
        x=dates, y=[row["opr_rate"] for row in series],
        mode="lines", name="OPR (%)", line=dict(color="#16a34a", width=2), yaxis="y2",
    ))
    fig_opr.update_layout(
        height=380, margin=dict(t=20, b=40, l=50, r=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="CPI Index"),
        yaxis2=dict(title="OPR (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_opr, use_container_width=True)

    st.divider()

    # ── HIES income snapshot ────────────────────────────────────────────────
    st.subheader(t["income_header"])
    income = fetch_income_snapshot(state)
    if income:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Median monthly income" if lang == "en" else "Pendapatan median bulanan", f"RM{income.get('income_median', 0):,.0f}")
        with col2:
            st.metric("Mean monthly expenditure" if lang == "en" else "Purata perbelanjaan bulanan", f"RM{income.get('expenditure_mean', 0):,.0f}")
        with col3:
            st.metric("Gini coefficient" if lang == "en" else "Pekali Gini", f"{income.get('gini', 0):.3f}")
        st.caption(
            "Source: DOSM Household Income & Expenditure Survey (HIES) 2022 — a static snapshot, not a monthly time series."
            if lang == "en" else
            "Sumber: Penyiasatan Pendapatan & Perbelanjaan Isi Rumah (HIES) DOSM 2022 — gambaran statik, bukan siri masa bulanan."
        )
    else:
        st.info("No income data available for this state." if lang == "en" else "Tiada data pendapatan untuk negeri ini.")

# ── main entry point ──────────────────────────────────────────────────────────
def main():
    state, spending, monthly_salary, horizon_months, mode, t, lang = render_sidebar()

    if mode.startswith("🏠"):
        render_home(t, lang)
    elif mode.startswith("📈"):
        render_data_explorer(t, lang, state)
    elif mode.startswith("📊"):
        render_personal_calculator(state, spending, monthly_salary, horizon_months, t, lang)
    else:
        render_relocation_planner(state, spending, monthly_salary, t, lang)

    st.markdown("---")
    footer_text = (
        "Malaysia Personal Inflation Intelligence | Data: DOSM & BNM via data.gov.my | "
        "Model: XGBoost trained on 2010–2026 CPI data | Built by Muhammad Ammar Ahmad Zaki"
        if lang == "en" else
        "Maklumat Inflasi Peribadi Malaysia | Data: DOSM & BNM melalui data.gov.my | "
        "Model: XGBoost dilatih atas data CPI 2010–2026 | Dibina oleh Muhammad Ammar Ahmad Zaki"
    )
    st.markdown(f"""
    <div style='text-align:center;color:#94a3b8;font-size:0.8rem;padding:10px'>
        {footer_text}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()