import os
import json
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score

DATA_PATH = "data/training/processed.csv"
MODEL_PATH = "models/model.joblib"
REPORT_DIR = "reports"
METRICS_JSON = os.path.join(REPORT_DIR, "metrics.json")
METRICS_MD = os.path.join(REPORT_DIR, "metrics.md")

TARGET = "target"

def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"No existe la columna target '{TARGET}' en {DATA_PATH}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ])

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "metric_primary": "ROC-AUC",
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "recall_benign_pos1": float(recall_score(y_test, pred, pos_label=1)),
        "recall_malignant_pos0": float(recall_score(y_test, pred, pos_label=0)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "model": "LogisticRegression + StandardScaler",
        "data": DATA_PATH,
        "test_size": 0.2,
        "random_state": 42,
        "n_features": int(X.shape[1]),
        "features": list(X.columns),
        "target_mapping": {"0": "malignant", "1": "benign"},
        "threshold": 0.5
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(model, MODEL_PATH)

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(METRICS_MD, "w", encoding="utf-8") as f:
        f.write("# Model Metrics (Breast Cancer)\n\n")
        f.write(f"- **Primary metric (ROC-AUC):** {metrics['roc_auc']:.5f}\n")
        f.write(f"- **F1:** {metrics['f1']:.5f}\n")
        f.write(f"- **Recall (benign=1):** {metrics['recall_benign_pos1']:.5f}\n")
        f.write(f"- **Recall (malignant=0):** {metrics['recall_malignant_pos0']:.5f}\n")
        f.write(f"- **Accuracy:** {metrics['accuracy']:.5f}\n\n")
        f.write(f"- Model: {metrics['model']}\n")
        f.write(f"- Dataset: `{DATA_PATH}`\n")
        f.write(f"- Features: {metrics['n_features']}\n")
        f.write(f"- Target mapping: 0=malignant, 1=benign\n")
        f.write(f"- Threshold: {metrics['threshold']}\n")

    print("=== Training completed ===")
    print("Model saved:", MODEL_PATH)
    print("Metrics saved:", METRICS_JSON, "and", METRICS_MD)
    print("ROC-AUC:", metrics["roc_auc"])

if __name__ == "__main__":
    main()