
import pandas as pd

df = pd.read_csv("data/processed/battery_soh.csv")

print("\nLOW CAPACITY MEASUREMENTS")
print(
    df[df["capacity_ah"] < 0.5][
        ["battery_id", "cycle_number", "capacity_ah", "soh_percent"]
    ].to_string(index=False)
)

print("\nCAPACITY BELOW 80% OF EACH BATTERY'S MAXIMUM")
df["relative_to_max"] = (
    df["capacity_ah"] /
    df.groupby("battery_id")["capacity_ah"].transform("max")
)

print(
    df[df["relative_to_max"] < 0.8][
        ["battery_id", "cycle_number", "capacity_ah", "relative_to_max"]
    ].to_string(index=False)
)

print("\nCAPACITY ORDER BY BATTERY")
for battery, group in df.groupby("battery_id"):
    print(f"\n{battery}")
    print(
        group.sort_values("capacity_ah")[
            ["cycle_number", "capacity_ah"]
        ].head(10).to_string(index=False)
    )
