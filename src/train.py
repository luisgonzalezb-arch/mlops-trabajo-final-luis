import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

DATA_PATH = Path("data") / "training" / "processed.csv"
MODEL_PATH = Path("models") / "model.joblib"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe {DATA_PATH}. Ejecuta src/data_preparation.py primero.")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42, n_estimators=200)
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"OK -> modelo guardado en {MODEL_PATH}")
    print(f"Features esperadas ({len(X.columns)}): {list(X.columns)}")

if __name__ == "__main__":
    main()
