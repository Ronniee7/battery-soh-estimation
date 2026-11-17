
from pathlib import Path
from scipy.io import loadmat

DATA_DIR = Path("data/raw")

for file_path in sorted(DATA_DIR.glob("*.mat")):
    print(f"\n{'=' * 55}")
    print(f"BATTERY: {file_path.stem}")
    print("=" * 55)

    data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    battery = data[file_path.stem]

    print("Battery structure:", type(battery))
    print("Available fields:", battery._fieldnames)

    cycles = battery.cycle

    print("Number of cycles:", len(cycles))

    for index, cycle in enumerate(cycles[:3]):
        print(f"\nCycle {index + 1}")
        print("Type:", cycle.type)
        print("Ambient temperature:", cycle.ambient_temperature)
        print("Data fields:", cycle.data._fieldnames)
