from flask import Flask, request, jsonify
from predict import predict_one, FEATURE_NAMES

app = Flask(__name__)

@app.get("/")
def health():
    return jsonify({"status": "ok", "features": FEATURE_NAMES})

@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        features = payload.get("features", payload)
        pred = predict_one(features)
        return jsonify({"prediction": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
