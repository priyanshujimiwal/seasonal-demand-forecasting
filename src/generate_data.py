"""
Generates a synthetic dataset of 3 years of daily demand (in kg) for
flowers, catering vegetables and fruits, with wedding-season and festival effects.

The data is simulated (not real market data) so the pipeline can be built and
tested end to end. Real mandi data from Agmarknet is planned for the next phase.
"""
import numpy as np
import pandas as pd
from features import festival_proximity_score, is_wedding_season

SEED = 42
START, END = "2023-01-01", "2025-12-31"

# base demand (kg/day), weekly effect, wedding uplift, festival uplift, noise level
CATEGORY_PROFILES = {
    "flowers":    dict(base=800,  weekend=0.10, wedding=0.45, festival=1.60, noise=0.08),
    "vegetables": dict(base=2500, weekend=0.06, wedding=0.30, festival=0.40, noise=0.06),
    "fruits":     dict(base=1500, weekend=0.08, wedding=0.20, festival=0.70, noise=0.07),
}


def generate(path="data/demand_data.csv"):
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, END, freq="D")
    prox = festival_proximity_score(dates)
    wedding = is_wedding_season(dates)
    day_of_year = dates.dayofyear.values
    weekend = (dates.dayofweek >= 5).astype(int)
    trend = np.linspace(1.0, 1.15, len(dates))          # slow growth over 3 years

    rows = []
    for cat, p in CATEGORY_PROFILES.items():
        yearly = 1 + 0.10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)   # mild seasonal wave
        demand = (p["base"] * trend * yearly
                  * (1 + p["weekend"] * weekend)
                  * (1 + p["wedding"] * wedding)
                  * (1 + p["festival"] * prox))
        demand *= rng.normal(1, p["noise"], len(dates))
        rows.append(pd.DataFrame({
            "date": dates,
            "category": cat,
            "demand_kg": demand.round(1),
            "is_wedding_season": wedding,
            "festival_proximity_score": prox,
            "day_of_week": dates.day_name(),
        }))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} rows to {path}")
    return df


if __name__ == "__main__":
    generate()
