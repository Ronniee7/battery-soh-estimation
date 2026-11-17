
from pathlib import Path
import pandas as pd

df = pd.read_csv("data/interim/discharge_cycles.csv")

# Remove zero and invalid capacity measurements
df = df[df["capacity_ah"] > 0].copy()

# Use maximum observed capacity per battery as provisional reference
df["reference_capacity_ah"] = (
    df.groupby("battery_id")["capacity_ah"].transform("max")
)

# Calculate capacity-based SOH
df["soh_percent"] = (
    df["capacity_ah"] / df["reference_capacity_ah"] * 100
)

output_file = Path("data/processed/battery_soh.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_file, index=False)

print("\nCorrected SOH calculation")
print("Valid discharge records:", len(df))
print("Excluded zero/invalid records:", 363 - len(df))

print("\nSOH summary:")
print(
    df.groupby("battery_id")["soh_percent"]
    .agg(["count", "mean", "min", "max"])
    .round(2)
)

print("\nReference capacities:")
print(
    df.groupby("battery_id")["reference_capacity_ah"]
    .first()
    .round(4)
)
