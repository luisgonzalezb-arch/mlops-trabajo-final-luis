from src.predict import predict_one

def test_predict_one_returns_expected_keys():
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

    out = predict_one(sample)
    assert "predicted_class" in out
    assert "probability_benign" in out
    assert "probability_malignant" in out