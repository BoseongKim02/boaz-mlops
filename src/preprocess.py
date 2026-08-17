# src/preprocess.py
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/Base.csv")
OUT_PATH = Path("data/processed/base.parquet")

# 결측치로 취급할 컬럼 (== -1 → NaN)
MINUS_ONE_TO_NAN = [
    "prev_address_months_count",
    "bank_months_count",
    "current_address_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

# 음수 전체를 결측으로 취급할 컬럼 (< 0 → NaN)
NEGATIVE_TO_NAN = ["intended_balcon_amount"]

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_uninformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    # device_fraud_count: 값이 전부 0 하나뿐 → 정보량 없음
    return df.drop(columns=["device_fraud_count"], errors="ignore")


def convert_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in MINUS_ONE_TO_NAN:
        df.loc[df[col] == -1, col] = pd.NA
    for col in NEGATIVE_TO_NAN:
        df.loc[df[col] < 0, col] = pd.NA
    # credit_risk_score, velocity_6h 음수는 정상값이므로 건드리지 않음
    return df


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def split_by_month(df: pd.DataFrame):
    train = df[df["month"].between(0, 5)]
    valid = df[df["month"] == 6]
    test = df[df["month"] == 7]
    return train, valid, test


def run():
    df = load_raw()
    df = drop_uninformative_columns(df)
    df = convert_missing_values(df)
    df = cast_categoricals(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    run()