# scripts/send_sample.py
import sys
import requests
import pandas as pd
from pathlib import Path

sys.path.append("src")
from preprocess import split_by_month, CATEGORICAL_COLS

TARGET = "fraud_bool"
URL = "http://localhost:8000/predict"


def main(n=3):
    df = pd.read_parquet(Path("data/processed/base.parquet"))

    # 6개월차(검증 구간)에서만 뽑는다. 7개월차는 봉인 구간.
    _, valid, _ = split_by_month(df)

    feature_cols = [c for c in df.columns if c not in [TARGET, "month"]]
    sample = valid[feature_cols].head(n)

    for i, (idx, row) in enumerate(sample.iterrows()):
        features = {}
        for col, val in row.items():
            if pd.isna(val):
                continue                      # 결측은 아예 안 보냄
            if col in CATEGORICAL_COLS:
                features[col] = str(val)      # 범주형은 문자열로
            else:
                features[col] = float(val)

        res = requests.post(URL, json={
            "application_id": f"valid-{idx}",
            "features": features,
        })
        print(f"[{i+1}] {res.status_code} {res.json()}")


if __name__ == "__main__":
    main()