"""Runs the full pipeline: generate data -> EDA -> train and evaluate models."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import generate_data, eda, forecast  # noqa: E402

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    print("Step 1/3: generating dataset");  generate_data.generate()
    print("\nStep 2/3: exploratory analysis"); eda.run()
    print("\nStep 3/3: training and evaluating models"); forecast.run()
    print("\nDone. Charts are in outputs/figures/ and scores in outputs/metrics.csv")
