
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 100
FEATURES = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
]

labels = pd.read_csv(OUTPUT_DIR / "battery_soh.csv")
labels = labels[labels["capacity_ah"] > 0]

X = []
y = []
battery_ids = []

for file_path in sorted(DATA_DIR.glob("*.mat")):
    battery_id = file_path.stem

    mat = loadmat(
        file_path,
        squeeze_me=True,
        struct_as_record=False,
    )
    cycles = mat[battery_id].cycle

    battery_labels = labels[
        labels["battery_id"] == battery_id
    ]

    soh_by_cycle = dict(
        zip(
            battery_labels["cycle_number"],
            battery_labels["soh_percent"],
        )
    )

    for cycle_number, cycle in enumerate(cycles, start=1):
        if cycle.type != "discharge":
            continue

        if cycle_number not in soh_by_cycle:
            continue

        measurements = cycle.data

        sequence = []

        for feature in FEATURES:
            values = np.asarray(
                getattr(measurements, feature),
                dtype=np.float32,
            ).reshape(-1)

            positions = np.linspace(0, 1, len(values))
            target_positions = np.linspace(
                0, 1, SEQUENCE_LENGTH
            )

            resampled = np.interp(
                target_positions,
                positions,
                values,
            )

            sequence.append(resampled)

        X.append(np.stack(sequence, axis=1))
        y.append(soh_by_cycle[cycle_number])
        battery_ids.append(battery_id)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)
battery_ids = np.asarray(battery_ids)

np.savez_compressed(
    OUTPUT_DIR / "sequence_dataset.npz",
    X=X,
    y=y,
    battery_ids=battery_ids,
)

print("Sequence dataset prepared")
print("Input shape:", X.shape)
print("Target shape:", y.shape)
print("Batteries:", np.unique(battery_ids, return_counts=True))
print("Saved:", OUTPUT_DIR / "sequence_dataset.npz")
