# src/train.py
from evaluate import evaluate
import mlflow
import lightgbm as lgb
import pandas as pd
from pathlib import Path
from preprocess import split_by_month, CATEGORICAL_COLS

DATA_PATH = Path("data/processed/base.parquet")
TARGET = "fraud_bool"
RANDOM_STATE = 42


def run():
    df = pd.read_parquet(DATA_PATH)
    train, valid, _ = split_by_month(df)

    feature_cols = [c for c in df.columns if c not in [TARGET, "month"]]
    X_train, y_train = train[feature_cols], train[TARGET]
    X_valid, y_valid = valid[feature_cols], valid[TARGET]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("baf-fraud-detection")
    mlflow.set_experiment("baf-fraud-detection")

    with mlflow.start_run(run_name="lgbm-baseline"):
        mlflow.log_param("model", "LightGBM")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("train_months", "0-5")
        mlflow.log_param("valid_month", "6")

        model = lgb.LGBMClassifier(random_state=RANDOM_STATE)
        model.fit(
            X_train, y_train,
            categorical_feature=CATEGORICAL_COLS,
        )

        model.booster_.save_model("model.txt")
        evaluate(model, X_valid, y_valid)
        mlflow.log_artifact("model.txt")

        return model, X_valid, y_valid


if __name__ == "__main__":
    run()