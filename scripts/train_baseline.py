
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/processed/model_dataset.csv")

features = [
    "cycle_number",
    "voltage_mean",
    "current_mean",
    "temperature_mean",
    "duration_seconds"
]

X = df[features]
y = df["soh_percent"]
groups = df["battery_id"]

logo = LeaveOneGroupOut()
results = []

print("\nLINEAR REGRESSION BASELINE")
print("=" * 45)

for train_idx, test_idx in logo.split(X, y, groups):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    battery = groups.iloc[test_idx].iloc[0]

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "battery_id": battery,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"\nTest battery: {battery}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

results_df = pd.DataFrame(results)

print("\nOVERALL RESULTS")
print("=" * 45)
print(results_df.round(4).to_string(index=False))

print("\nMean performance across batteries:")
print(results_df[["MAE", "RMSE", "R2"]].mean().round(4))

results_df.to_csv(
    "outputs/tables/baseline_results.csv",
    index=False
)

print("\nResults saved to outputs/tables/baseline_results.csv")
