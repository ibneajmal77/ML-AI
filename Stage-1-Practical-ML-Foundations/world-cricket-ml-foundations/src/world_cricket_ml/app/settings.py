from __future__ import annotations

from datetime import date, timedelta


TOP_20_T20I_TEAMS = [
    "India",
    "England",
    "Australia",
    "New Zealand",
    "South Africa",
    "Pakistan",
    "West Indies",
    "Sri Lanka",
    "Bangladesh",
    "Afghanistan",
    "Zimbabwe",
    "Ireland",
    "Netherlands",
    "Scotland",
    "Namibia",
    "USA",
    "United Arab Emirates",
    "Nepal",
    "Canada",
    "Oman",
]

ROLLING_HISTORY_QUARTERS = 8
REFRESH_CADENCE_MONTHS = 3


def rolling_two_year_quarterly_window(reference: date | None = None) -> tuple[date, date]:
    """Return the last eight completed quarters as a closed date window.

    This keeps roughly two years of history and advances on a quarterly cadence.
    For example, on 2026-04-07 the window is 2024-04-01 to 2026-03-31.
    """
    today = reference or date.today()
    current_quarter_start_month = ((today.month - 1) // 3) * 3 + 1
    current_quarter_start = date(today.year, current_quarter_start_month, 1)
    end_date = current_quarter_start - timedelta(days=1)

    start_month = current_quarter_start.month
    start_year = current_quarter_start.year
    for _ in range(ROLLING_HISTORY_QUARTERS):
        start_month -= 3
        if start_month <= 0:
            start_month += 12
            start_year -= 1
    start_date = date(start_year, start_month, 1)
    return start_date, end_date
