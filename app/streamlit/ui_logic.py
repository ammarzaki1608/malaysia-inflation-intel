"""
Small pieces of the app's page logic that need no Streamlit, so they can be
unit-tested with the standard library only (see test_ui_logic.py).
"""

import json

MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_MS = ["Jan", "Feb", "Mac", "Apr", "Mei", "Jun", "Jul", "Ogos", "Sep", "Okt", "Nov", "Dis"]


def month_label(iso_date, lang: str) -> str:
    """'2026-09-01' or '2026-09-01 00:00:00' -> 'Sep 2026'. Empty if the date can't be read."""
    try:
        year, month = int(str(iso_date)[:4]), int(str(iso_date)[5:7])
    except (ValueError, TypeError):
        return ""
    if not 1 <= month <= 12:          # month 0 would otherwise wrap round to December
        return ""
    return f"{(MONTHS_EN if lang == 'en' else MONTHS_MS)[month - 1]} {year}"


def insight_signature(summary_data: dict) -> str:
    """Identifies the numbers an AI suggestion was written from."""
    return json.dumps(summary_data, sort_keys=True, default=str)


def saved_insight(saved, signature: str):
    """
    The suggestion text kept in session state, but only if it was written from
    these same numbers; otherwise None, so the user is offered a new one.
    """
    if saved and saved.get("signature") == signature and saved.get("text"):
        return saved["text"]
    return None
