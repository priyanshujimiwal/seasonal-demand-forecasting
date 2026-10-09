"""Exploratory data analysis: produces 4 charts in outputs/figures/."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "outputs/figures"


def run(path="data/demand_data.csv"):
    df = pd.read_csv(path, parse_dates=["date"])

    # 1. Daily demand over time for each category
    fig, ax = plt.subplots(figsize=(12, 5))
    for cat, g in df.groupby("category"):
        ax.plot(g["date"], g["demand_kg"], label=cat, linewidth=0.8)
    ax.set(title="Daily demand by category", xlabel="Date", ylabel="Demand (kg)")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/eda_1_demand_over_time.png", dpi=120); plt.close(fig)

    # 2. Average demand by month
    monthly = df.assign(month=df["date"].dt.month).groupby(["month", "category"])["demand_kg"].mean().unstack()
    ax = monthly.plot(kind="bar", figsize=(10, 5), title="Average daily demand by month")
    ax.set(xlabel="Month", ylabel="Demand (kg)")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eda_2_monthly_average.png", dpi=120); plt.close()

    # 3. Wedding season vs normal days
    ws = df.groupby(["category", "is_wedding_season"])["demand_kg"].mean().unstack()
    ws.columns = ["Normal", "Wedding season"]
    ax = ws.plot(kind="bar", figsize=(8, 5), title="Average demand: wedding season vs normal")
    ax.set(xlabel="Category", ylabel="Demand (kg)")
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/eda_3_wedding_season.png", dpi=120); plt.close()

    # 4. Festival proximity vs demand (flowers, the most festival-driven category)
    fl = df[df["category"] == "flowers"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fl["festival_proximity_score"], fl["demand_kg"], alpha=0.4, s=12)
    ax.set(title="Flowers: demand vs festival proximity score",
           xlabel="Festival proximity score (0 = no festival, 1 = festival day)", ylabel="Demand (kg)")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/eda_4_festival_proximity.png", dpi=120); plt.close(fig)

    print(f"Saved 4 EDA charts to {FIG_DIR}/")


if __name__ == "__main__":
    run()
