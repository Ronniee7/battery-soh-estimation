
from pathlib import Path
import pandas as pd
from scipy.io import loadmat

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/interim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []

for file_path in sorted(DATA_DIR.glob("*.mat")):
    battery_id = file_path.stem

    mat = loadmat(
        file_path,
        squeeze_me=True,
        struct_as_record=False
    )

    cycles = mat[battery_id].cycle

    for cycle_number, cycle in enumerate(cycles, start=1):

        if cycle.type != "discharge":
            continue

        measurements = cycle.data

        capacity = float(measurements.Capacity)

        records.append({
            "battery_id": battery_id,
            "cycle_number": cycle_number,
            "ambient_temperature": float(cycle.ambient_temperature),
            "capacity_ah": capacity,
            "voltage_mean": measurements.Voltage_measured.mean(),
            "current_mean": measurements.Current_measured.mean(),
            "temperature_mean": measurements.Temperature_measured.mean(),
            "duration_seconds": (
                measurements.Time[-1] - measurements.Time[0]
            )
        })

df = pd.DataFrame(records)

output_file = OUTPUT_DIR / "discharge_cycles.csv"
df.to_csv(output_file, index=False)

print("\nDischarge extraction complete")
print("Total discharge cycles:", len(df))
print("\nCycles per battery:")
print(df.groupby("battery_id").size())

print("\nCapacity summary:")
print(df.groupby("battery_id")["capacity_ah"].describe())

print(f"\nSaved to: {output_file}")
