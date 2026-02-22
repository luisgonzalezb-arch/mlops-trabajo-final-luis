import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

RAW_PATH = "data/raw/breast_cancer.csv"
PROCESSED_PATH = "data/training/processed.csv"

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/training", exist_ok=True)

    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()  # incluye features + target

    # Guardar RAW (tal cual)
    df.to_csv(RAW_PATH, index=False)

    # Guardar processed (en este caso, igual al raw; dataset ya viene limpio)
    df.to_csv(PROCESSED_PATH, index=False)

    print("=== Data preparation completed ===")
    print("Raw:", RAW_PATH)
    print("Processed:", PROCESSED_PATH)
    print("target_names:", list(data.target_names))
    print("Target mapping: 0=malignant, 1=benign")

if __name__ == "__main__":
    main()