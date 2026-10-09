"""
Trains and compares two Prophet models for each product category:
  - Baseline: only the date (ds) and demand (y)
  - Improved: adds festival_proximity_score as an external regressor
Both are evaluated on the last 90 days, held out as a test set.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prophet import Prophet

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True

TEST_DAYS = 90
FIG_DIR = "outputs/figures"


def mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))


def mape(actual, pred):
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def fit_predict(train, test, use_festival):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    cols = ["ds", "y"]
    if use_festival:
        model.add_regressor("festival_proximity_score")
        cols.append("festival_proximity_score")
    model.fit(train[cols])
    future = test[[c for c in cols if c != "y"]]
    return model.predict(future)["yhat"].values


def plot_forecast(cat, history, test, base_pred, imp_pred):
    fig, ax = plt.subplots(figsize=(12, 5))
    recent = history.tail(120)
    ax.plot(recent["ds"], recent["y"], color="grey", linewidth=0.8, label="Training data (last 120 days)")
    ax.scatter(test["ds"], test["y"], color="black", s=10, label="Actual (test)")
    ax.plot(test["ds"], base_pred, color="tab:orange", label="Baseline Prophet")
    ax.plot(test["ds"], imp_pred, color="tab:blue", label="Prophet + festival score")
    ax.set(title=f"{cat.capitalize()}: 90-day forecast vs actual", xlabel="Date", ylabel="Demand (kg)")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/forecast_{cat}.png", dpi=120); plt.close(fig)


def run(path="data/demand_data.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    results = []
    for cat, g in df.groupby("category"):
        g = g.rename(columns={"date": "ds", "demand_kg": "y"}).sort_values("ds")
        train, test = g.iloc[:-TEST_DAYS], g.iloc[-TEST_DAYS:]
        base = fit_predict(train, test, use_festival=False)
        imp = fit_predict(train, test, use_festival=True)
        actual = test["y"].values
        results += [
            {"category": cat, "model": "Baseline Prophet", "MAE_kg": mae(actual, base), "MAPE_%": mape(actual, base)},
            {"category": cat, "model": "Prophet + festival score", "MAE_kg": mae(actual, imp), "MAPE_%": mape(actual, imp)},
        ]
        plot_forecast(cat, train, test, base, imp)
        print(f"Finished {cat}")

    res = pd.DataFrame(results).round(2)
    res.to_csv("outputs/metrics.csv", index=False)
    print("\n", res.to_string(index=False))
    return res


if __name__ == "__main__":
    run()
