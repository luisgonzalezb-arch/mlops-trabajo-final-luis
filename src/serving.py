from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

MODEL_PATH = "models/model.joblib"

app = Flask(__name__)
model = load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Breast Cancer classifier API"})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    # Espera: {"features": {...}} o directamente {...}
    features = payload.get("features", payload)

    X = pd.DataFrame([features])
    proba_benign = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba_benign >= 0.5)

    return jsonify({
        "predicted_class": pred,
        "probability_benign": proba_benign,
        "probability_malignant": 1 - proba_benign,
        "mapping": { "0": "malignant", "1": "benign" },
        "threshold": 0.5
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)