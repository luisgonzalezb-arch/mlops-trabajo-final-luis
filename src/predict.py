import pandas as pd
from joblib import load

MODEL_PATH = "models/model.joblib"

def predict_one(features: dict):
    model = load(MODEL_PATH)
    X = pd.DataFrame([features])
    proba = float(model.predict_proba(X)[:, 1][0])  # prob de clase 1 (benign)
    pred = int(proba >= 0.5)
    return {
        "predicted_class": pred,
        "probability_benign": proba,
        "probability_malignant": 1 - proba,
        "mapping": {0: "malignant", 1: "benign"},
        "threshold": 0.5
    }

if __name__ == "__main__":
    sample = {
        "mean radius": 14.0,
        "mean texture": 19.0,
        "mean perimeter": 90.0,
        "mean area": 600.0,
        "mean smoothness": 0.1,
        "mean compactness": 0.1,
        "mean concavity": 0.1,
        "mean concave points": 0.05,
        "mean symmetry": 0.18,
        "mean fractal dimension": 0.06,
        "radius error": 0.5,
        "texture error": 1.0,
        "perimeter error": 3.0,
        "area error": 40.0,
        "smoothness error": 0.01,
        "compactness error": 0.02,
        "concavity error": 0.03,
        "concave points error": 0.01,
        "symmetry error": 0.02,
        "fractal dimension error": 0.003,
        "worst radius": 16.0,
        "worst texture": 25.0,
        "worst perimeter": 105.0,
        "worst area": 850.0,
        "worst smoothness": 0.14,
        "worst compactness": 0.25,
        "worst concavity": 0.3,
        "worst concave points": 0.1,
        "worst symmetry": 0.3,
        "worst fractal dimension": 0.08
    }

    print(predict_one(sample))