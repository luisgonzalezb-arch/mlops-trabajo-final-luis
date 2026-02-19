from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models") / "model.joblib"

FEATURE_NAMES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python src/train.py")
    return joblib.load(MODEL_PATH)

def predict_one(features: dict) -> float:
    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        raise ValueError(f"Missing fields: {missing}. Expected: {FEATURE_NAMES}")

    x = pd.DataFrame([{f: float(features[f]) for f in FEATURE_NAMES}])
    model = load_model()
    pred = float(model.predict(x)[0])
    return pred
