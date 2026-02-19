from src.predict import predict_one

def test_predict_returns_float():
    x = {"age":0.05,"sex":0.02,"bmi":0.04,"bp":0.01,"s1":0.03,"s2":-0.02,"s3":-0.01,"s4":0.02,"s5":0.04,"s6":0.01}
    y = predict_one(x)
    assert isinstance(y, float)
