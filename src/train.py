# src/train.py
import os
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

DATA_PATH = "data/training/processed.csv"
MODEL_PATH = "models/model.joblib"
REPORT_DIR = "reports"
METRICS_JSON = os.path.join(REPORT_DIR, "metrics.json")
METRICS_MD = os.path.join(REPORT_DIR, "metrics.md")

TARGET = "y"

def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"No existe la columna target '{TARGET}' en {DATA_PATH}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Console output
    print("=== Training completed ===")
    print(f"Features ({len(X.columns)}): {list(X.columns)}")
    print(f"RMSE (test): {rmse:.4f}")

    # Persist artifacts
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(model, MODEL_PATH)

    os.makedirs(REPORT_DIR, exist_ok=True)

    metrics = {
        "metric": "RMSE",
        "rmse_test": rmse,
        "model": "RandomForestRegressor",
        "data": DATA_PATH,
        "test_size": 0.2,
        "random_state": 42,
        "n_features": int(X.shape[1]),
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(METRICS_MD, "w", encoding="utf-8") as f:
        f.write("# Model Metrics\n\n")
        f.write(f"- **Metric:** RMSE\n")
        f.write(f"- **RMSE (test):** {rmse:.4f}\n")
        f.write(f"- **Model:** RandomForestRegressor\n")
        f.write(f"- **Dataset:** `{DATA_PATH}`\n")
        f.write(f"- **Test split:** 0.2\n")
        f.write(f"- **Random state:** 42\n")

if __name__ == "__main__":
    main()