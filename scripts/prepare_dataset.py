
from pathlib import Path
import pandas as pd

df = pd.read_csv("data/processed/battery_soh.csv")

features = [
    "cycle_number",
    "voltage_mean",
    "current_mean",
    "temperature_mean",
    "duration_seconds"
]

target = "soh_percent"

dataset = df[
    ["battery_id"] + features + [target]
].copy()

dataset = dataset.dropna()

output = Path("data/processed/model_dataset.csv")
dataset.to_csv(output, index=False)

print("\nModel dataset prepared")
print("Shape:", dataset.shape)

print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)

print("\nRecords per battery:")
print(dataset.groupby("battery_id").size())

print("\nMissing values:")
print(dataset.isna().sum())

print(f"\nSaved to: {output}")
