
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load experiment results
baseline = pd.read_csv(TABLE_DIR / "baseline_results.csv")
cnn = pd.read_csv(TABLE_DIR / "cnn_sequence_results.csv")
lstm = pd.read_csv(TABLE_DIR / "lstm_sequence_results.csv")

# Identify the models
baseline["model"] = "Linear Regression"
cnn["model"] = "1D CNN"
lstm["model"] = "LSTM"

# Combine all results
all_results = pd.concat(
    [baseline, cnn, lstm],
    ignore_index=True
)

metrics = ["MAE", "RMSE", "R2"]

# Summary statistics across the four held-out batteries
summary = (
    all_results.groupby("model")[metrics + ["training_time_s"]]
    .agg(["mean", "std"])
    .round(4)
)

summary.to_csv(TABLE_DIR / "final_model_comparison.csv")

# Per-battery comparison
per_battery = all_results[
    ["battery_id", "model", "MAE", "RMSE", "R2", "training_time_s"]
].copy()

per_battery.to_csv(
    TABLE_DIR / "final_per_battery_results.csv",
    index=False
)

# Print report tables
print("\nFINAL MODEL COMPARISON")
print(
    all_results.groupby("model")[
        metrics + ["training_time_s"]
    ].mean().round(4).to_string()
)

print("\nPER-BATTERY RESULTS")
print(
    per_battery.pivot(
        index="battery_id",
        columns="model",
        values="MAE"
    ).round(4).to_string()
)

# Plot 1: Mean MAE
mean_metrics = all_results.groupby("model")[metrics].mean()

plt.figure(figsize=(8, 5))
mean_metrics["MAE"].sort_values().plot(
    kind="bar",
    rot=0
)
plt.ylabel("Mean Absolute Error (percentage points)")
plt.xlabel("Model")
plt.title("Mean MAE Across Held-Out Batteries")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "mean_mae_comparison.png", dpi=300)
plt.close()

# Plot 2: Mean RMSE
plt.figure(figsize=(8, 5))
mean_metrics["RMSE"].sort_values().plot(
    kind="bar",
    rot=0
)
plt.ylabel("Root Mean Squared Error (percentage points)")
plt.xlabel("Model")
plt.title("Mean RMSE Across Held-Out Batteries")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "mean_rmse_comparison.png", dpi=300)
plt.close()

# Plot 3: R2 comparison
plt.figure(figsize=(8, 5))
mean_metrics["R2"].plot(kind="bar", rot=0)
plt.axhline(0, linestyle="--")
plt.ylabel("Mean R²")
plt.xlabel("Model")
plt.title("Mean R² Across Held-Out Batteries")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "mean_r2_comparison.png", dpi=300)
plt.close()

# Plot 4: MAE for each held-out battery
mae_pivot = per_battery.pivot(
    index="battery_id",
    columns="model",
    values="MAE"
)

mae_pivot.plot(kind="bar", figsize=(9, 5))
plt.ylabel("MAE (percentage points)")
plt.xlabel("Held-Out Battery")
plt.title("Model MAE by Test Battery")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "mae_by_battery.png", dpi=300)
plt.close()

# Plot 5: Mean training time
training_times = all_results.groupby("model")["training_time_s"].mean()

plt.figure(figsize=(8, 5))
training_times.plot(kind="bar", rot=0)
plt.ylabel("Mean Training Time (seconds)")
plt.xlabel("Model")
plt.title("Average Model Training Time")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "training_time_comparison.png", dpi=300)
plt.close()

print("\nFINAL RESULTS GENERATED")
print("Tables:")
print(" - outputs/tables/final_model_comparison.csv")
print(" - outputs/tables/final_per_battery_results.csv")

print("\nFigures:")
for filename in [
    "mean_mae_comparison.png",
    "mean_rmse_comparison.png",
    "mean_r2_comparison.png",
    "mae_by_battery.png",
    "training_time_comparison.png"
]:
    print(" - outputs/figures/" + filename)
