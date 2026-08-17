# tests/conftest.py
import os
import numpy as np
import pandas as pd
import pytest
import mlflow
import lightgbm as lgb
from mlflow import MlflowClient
from mlflow.models import infer_signature

CAT_COL = "device_os"
CATS = ["linux", "windows", "macintosh"]


@pytest.fixture(scope="session")
def api_client(tmp_path_factory):
    """가짜 모델을 레지스트리에 올린 뒤, 그걸 쓰는 API 클라이언트를 만든다."""
    root = tmp_path_factory.mktemp("mlops")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{root}/mlflow.db"
    os.environ["PREDICTION_LOG_PATH"] = str(root / "predictions.jsonl")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # ① 가짜 학습 데이터
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "income": rng.normal(size=200),
        "credit_risk_score": rng.integers(0, 300, size=200).astype(float),
        CAT_COL: pd.Categorical(rng.choice(CATS, 200), categories=CATS),
    })
    y = (rng.random(200) < 0.2).astype(int)

    model = lgb.LGBMClassifier(n_estimators=5, random_state=42, verbose=-1)
    model.fit(X, y, categorical_feature=[CAT_COL])

    # ② 실제 train.py와 같은 방식으로 기록
    mlflow.set_experiment("test-exp")
    with mlflow.start_run():
        mlflow.log_metric("threshold_at_fpr5pct", 0.5)
        mlflow.log_dict({CAT_COL: CATS}, "categories.json")

        sample = X.head(3).copy()
        sample[CAT_COL] = sample[CAT_COL].astype(str)
        sig = infer_signature(sample, model.predict_proba(X.head(3))[:, 1])

        info = mlflow.lightgbm.log_model(
            model, name="model", signature=sig,
            registered_model_name="fraud-detector",
        )

    MlflowClient().set_registered_model_alias(
        "fraud-detector", "champion", info.registered_model_version
    )

    # ③ 환경변수 설정 후에 import (순서 중요)
    from fastapi.testclient import TestClient
    from serving.app import app
    return TestClient(app)