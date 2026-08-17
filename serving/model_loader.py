# serving/model_loader.py
import os
import json
import mlflow
from mlflow import MlflowClient

MODEL_NAME = "fraud-detector"
ALIAS = "champion"


def load_bundle():
    """champion 별칭이 가리키는 모델과, 그 모델에 딸린 임계값·카테고리를 함께 꺼낸다."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = MlflowClient()

    # ① 별칭 → 모델 버전
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

    # ② 모델 버전 → 그 모델을 만든 run
    run = client.get_run(mv.run_id)

    # ③ run에서 임계값을 꺼낸다 (코드에 상수 없음)
    threshold = run.data.metrics["threshold_at_fpr5pct"]

    # ④ run에서 카테고리 순서를 꺼낸다
    path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="categories.json"
    )
    with open(path, encoding="utf-8") as f:
        categories = json.load(f)

    # ⑤ 모델 본체 (predict_proba가 살아있는 객체로)
    model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

    return {
        "model": model,
        "threshold": threshold,
        "categories": categories,
        "model_version": f"{MODEL_NAME}:{mv.version}",
        "feature_names": list(model.feature_name_),
    }