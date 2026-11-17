
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/battery_soh.csv")

plt.figure(figsize=(10, 6))

for battery, group in df.groupby("battery_id"):
    group = group.sort_values("cycle_number")

    plt.plot(
        group["cycle_number"],
        group["soh_percent"],
        marker="o",
        markersize=3,
        label=battery
    )

plt.xlabel("Cycle Number")
plt.ylabel("State of Health (%)")
plt.title("Battery State of Health Degradation")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output = Path("outputs/figures/soh_degradation.png")
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=300)
plt.show()

print(f"Plot saved to: {output}")
