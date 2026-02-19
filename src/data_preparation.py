import pandas as pd
from pathlib import Path
from sklearn.datasets import load_diabetes

RAW_PATH = Path("data") / "raw" / "diabetes.csv"
OUT_PATH = Path("data") / "training" / "processed.csv"

def main():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ds = load_diabetes(as_frame=True)
    df = ds.frame  # incluye features + target
    df.rename(columns={"target": "y"}, inplace=True)

    df.to_csv(RAW_PATH, index=False)
    df.to_csv(OUT_PATH, index=False)

    print(f"OK -> raw: {RAW_PATH} | processed: {OUT_PATH} | rows={len(df)} | cols={df.shape[1]}")

if __name__ == "__main__":
    main()
