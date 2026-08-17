# src/train.py
from mlflow.models import infer_signature
from mlflow import MlflowClient
from evaluate import evaluate
import mlflow
import lightgbm as lgb
import pandas as pd
from pathlib import Path
from preprocess import split_by_month, CATEGORICAL_COLS

DATA_PATH = Path("data/processed/base.parquet")
TARGET = "fraud_bool"
RANDOM_STATE = 42
MODEL_NAME = "fraud-detector"


def run():
    df = pd.read_parquet(DATA_PATH)
    train, valid, _ = split_by_month(df)

    feature_cols = [c for c in df.columns if c not in [TARGET, "month"]]
    X_train, y_train = train[feature_cols], train[TARGET]
    X_valid, y_valid = valid[feature_cols], valid[TARGET]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
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

        evaluate(model, X_valid, y_valid)

        # 카테고리 순서를 박제 (서빙에서 동일하게 복원해야 함)
        categories = {
            col: [str(v) for v in X_train[col].cat.categories]
            for col in CATEGORICAL_COLS
        }
        mlflow.log_dict(categories, "categories.json")

        # signature: 범주형은 문자열로 선언 (API 입력이 문자열이므로)
        sample = X_valid.head(5).copy()
        for col in CATEGORICAL_COLS:
            sample[col] = sample[col].astype(str)
        signature = infer_signature(
            sample, model.predict_proba(X_valid.head(5))[:, 1]
        )

        model_info = mlflow.lightgbm.log_model(
            model,
            name="model",
            signature=signature,
            input_example=sample,
            registered_model_name=MODEL_NAME,
        )

    # champion 별칭 붙이기 (run 블록 밖)
    client = MlflowClient()
    client.set_registered_model_alias(
        MODEL_NAME, "champion", model_info.registered_model_version
    )
    print(f"champion → {MODEL_NAME} 버전 {model_info.registered_model_version}")

    return model, X_valid, y_valid


if __name__ == "__main__":
    run()