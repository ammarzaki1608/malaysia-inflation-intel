"""Every BigQuery query the API runs. One shared client, created on first use."""

from google.cloud import bigquery

from service.config import PROJECT_ID

_client = None


def client():
    global _client
    if _client is None:
        _client = bigquery.Client(project=PROJECT_ID)
    return _client


def _query(sql: str, params: list | None = None) -> list:
    job_config = bigquery.QueryJobConfig(query_parameters=params or [])
    return list(client().query(sql, job_config=job_config).result())


def feature_history(state: str, division: str, months_back: int = 15) -> list[dict]:
    """
    The last N months of real CPI, PPI and OPR for one series, oldest first.
    Used by /trends.
    """
    rows = _query(
        f"""
        select
            date, cpi_index,
            cpi_lag_1, cpi_lag_3, cpi_lag_6, cpi_lag_12,
            cpi_ma_3, cpi_ma_12,
            ppi_index, ppi_lag_1, ppi_lag_2, ppi_lag_3,
            opr_rate
        from `{PROJECT_ID}.dbt_.mart_cpi_features`
        where state = @state
            and division = @division
            and cpi_lag_1 is not null
        order by date desc
        limit @months_back
        """,
        [
            bigquery.ScalarQueryParameter("state", "STRING", state),
            bigquery.ScalarQueryParameter("division", "STRING", division),
            bigquery.ScalarQueryParameter("months_back", "INT64", months_back),
        ],
    )
    history = [dict(row) for row in rows]
    history.reverse()

    # Forward-fill OPR nulls — BNM publishes with a 1-2 month lag,
    # so the most recent months are often null. Carry the last
    # known real rate forward, matching the training preprocessing.
    last_known_opr = None
    for row in history:
        if row.get("opr_rate") is not None:
            last_known_opr = row["opr_rate"]
        else:
            row["opr_rate"] = last_known_opr

    return history


def income_snapshot(state: str) -> dict | None:
    """The HIES income and expenditure snapshot for one state, or None."""
    rows = _query(
        f"""
        select
            income_median, income_mean, expenditure_mean,
            gini, poverty_rate, income_band
        from `{PROJECT_ID}.dbt_.mart_income_bands`
        where state = @state
        limit 1
        """,
        [bigquery.ScalarQueryParameter("state", "STRING", state)],
    )
    return dict(rows[0]) if rows else None


def national_cpi_by_division() -> dict:
    """{division: [rows, newest first]} from the national CPI table, without the overall index."""
    rows = _query(
        f"""
        select division, date, index
        from `{PROJECT_ID}.raw.cpi_national`
        where division != 'overall'
        order by division, date desc
        """
    )
    by_division: dict = {}
    for row in rows:
        by_division.setdefault(row["division"], []).append(row)
    return by_division


def national_headline_cpi() -> list[dict]:
    """The last 13 months of the national overall CPI, newest first."""
    rows = _query(
        f"""
        select date, index
        from `{PROJECT_ID}.raw.cpi_national`
        where division = 'overall'
        order by date desc
        limit 13
        """
    )
    return [dict(r) for r in rows]


def latest_cpi_by_state(division: str) -> list[dict]:
    """The newest CPI row of every state for one category."""
    rows = _query(
        f"""
        select state, cpi_index, cpi_lag_12, date
        from `{PROJECT_ID}.dbt_.mart_cpi_features`
        where division = @division
          and cpi_index is not null
        qualify row_number() over (partition by state order by date desc) = 1
        """,
        [bigquery.ScalarQueryParameter("division", "STRING", division)],
    )
    return [dict(r) for r in rows]
