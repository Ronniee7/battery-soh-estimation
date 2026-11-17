
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

from src.models.lstm import LSTMRegressor

torch.manual_seed(42)
np.random.seed(42)

data = np.load("data/processed/sequence_dataset.npz")

X = data["X"].astype(np.float32)
y = data["y"].astype(np.float32)
groups = data["battery_ids"]

results = []
logo = LeaveOneGroupOut()

for train_idx, test_idx in logo.split(X, y, groups):

    test_battery = groups[test_idx][0]

    # Scale input features using training batteries only
    x_scaler = StandardScaler()

    X_train = X[train_idx]
    X_test = X[test_idx]

    X_train = x_scaler.fit_transform(
        X_train.reshape(-1, 3)
    ).reshape(X_train.shape)

    X_test = x_scaler.transform(
        X_test.reshape(-1, 3)
    ).reshape(X_test.shape)

    # Standardise targets using training batteries only
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(
        y[train_idx].reshape(-1, 1)
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(
        y_train_scaled, dtype=torch.float32
    )

    model = LSTMRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    start = time.time()

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train_tensor)
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
        "battery_id": str(test_battery),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "training_time_s": training_time
    })

    print(f"\nTest battery: {test_battery}")
    print(results[-1])

results_df = pd.DataFrame(results)

Path("outputs/tables").mkdir(parents=True, exist_ok=True)
results_df.to_csv(
    "outputs/tables/lstm_sequence_results.csv",
    index=False
)

print("\nLSTM RESULTS")
print(results_df.round(4).to_string(index=False))

print("\nMean performance:")
print(results_df[["MAE", "RMSE", "R2"]].mean().round(4))

print("\nSaved: outputs/tables/lstm_sequence_results.csv")
