"""Calendar features: Indian festival dates, wedding seasons and the festival proximity score."""
import pandas as pd

# Major festival dates (2023 to 2026). Dates are approximate and follow the
# commonly published Indian calendar; update them if your source differs.
FESTIVALS = {
    "Diwali":           ["2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08"],
    "Holi":             ["2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04"],
    "Navratri":         ["2023-10-15", "2024-10-03", "2025-09-22", "2026-10-11"],
    "Durga Puja":       ["2023-10-24", "2024-10-12", "2025-10-02", "2026-10-20"],
    "Ganesh Chaturthi": ["2023-09-19", "2024-09-07", "2025-08-27", "2026-09-14"],
    "Eid":              ["2023-04-22", "2024-04-11", "2025-03-31", "2026-03-21"],
}

FESTIVAL_WINDOW = 4  # days before and after a festival that feel its effect

# Indian wedding seasons: November to February, and April to June
WEDDING_MONTHS = {11, 12, 1, 2, 4, 5, 6}


def festival_dates():
    """All festival dates as a sorted list of Timestamps."""
    return sorted(pd.Timestamp(d) for dates in FESTIVALS.values() for d in dates)


def festival_proximity_score(dates):
    """
    Score between 0 and 1 for each date.
    1.0 on the festival day, decaying linearly to 0.2 at 4 days away,
    and 0 outside the +/- 4 day window. If two festivals overlap, the higher score wins.
    """
    dates = pd.to_datetime(pd.Series(dates))
    fests = festival_dates()
    def score(day):
        best = 0.0
        for f in fests:
            gap = abs((day - f).days)
            if gap <= FESTIVAL_WINDOW:
                best = max(best, 1 - gap / (FESTIVAL_WINDOW + 1))
        return round(best, 3)
    return dates.apply(score).values


def is_wedding_season(dates):
    """1 if the date falls in a wedding-season month, else 0."""
    return pd.to_datetime(pd.Series(dates)).dt.month.isin(WEDDING_MONTHS).astype(int).values
