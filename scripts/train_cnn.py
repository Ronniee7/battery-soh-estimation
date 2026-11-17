
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn import CNNRegressor

torch.manual_seed(42)
np.random.seed(42)

# Load sequence dataset
data = np.load("data/processed/sequence_dataset.npz")

X = data["X"].astype(np.float32)
y = data["y"].astype(np.float32)
groups = data["battery_ids"]

results = []
logo = LeaveOneGroupOut()

for train_idx, test_idx in logo.split(X, y, groups):

    test_battery = groups[test_idx][0]

    # Scale using training data only
    scaler = StandardScaler()
    X_train = X[train_idx]
    X_test = X[test_idx]

    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)

    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)

    # CNN expects (samples, channels, sequence_length)
    X_train = torch.tensor(X_train.transpose(0, 2, 1))
    X_test = torch.tensor(X_test.transpose(0, 2, 1))

    y_scaler = StandardScaler()

    y_train_scaled = y_scaler.fit_transform(
        y[train_idx].reshape(-1, 1)
    )

    y_train = torch.tensor(
        y_train_scaled, dtype=torch.float32
    )

    model = CNNRegressor(input_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    start = time.time()

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

    training_time = time.time() - start

    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test).numpy()

    predictions = y_scaler.inverse_transform(
    predictions_scaled
    ).flatten()

    actual = y[test_idx]

    mae = np.mean(np.abs(actual - predictions))
    rmse = np.sqrt(np.mean((actual - predictions) ** 2))
    r2 = 1 - (
        np.sum((actual - predictions) ** 2) /
        np.sum((actual - actual.mean()) ** 2)
    )

    results.append({
        "battery_id": test_battery,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "training_time_s": training_time
    })

    print(f"\nTest battery: {test_battery}")
    print(results[-1])

results_df = pd.DataFrame(results)

Path("outputs/tables").mkdir(parents=True, exist_ok=True)
results_df.to_csv("outputs/tables/cnn_sequence_results.csv", index=False)

print("\nSEQUENCE CNN RESULTS")
print(results_df.round(4).to_string(index=False))

print("\nMean performance:")
print(results_df[["MAE", "RMSE", "R2"]].mean().round(4))

print("\nSaved: outputs/tables/cnn_sequence_results.csv")
