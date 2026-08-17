# serving/app.py
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from serving.model_loader import load_bundle
from serving.prediction_log import log_prediction

app = FastAPI(title="Fraud Detection API")

# 서버가 뜰 때 딱 한 번만 로드
BUNDLE = load_bundle()


class PredictRequest(BaseModel):
    application_id: str
    features: dict


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": BUNDLE["model_version"],
        "threshold": BUNDLE["threshold"],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.features])

    # ① 학습 때와 같은 컬럼 순서로 정렬 (없는 컬럼은 NaN)
    df = df.reindex(columns=BUNDLE["feature_names"])

    # ② 문자열 → 학습 때와 같은 순서의 범주형으로 복원
    for col, cats in BUNDLE["categories"].items():
        df[col] = pd.Categorical(df[col].astype("string"), categories=cats)

    # ③ 판정
    try:
        score = float(BUNDLE["model"].predict_proba(df)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"예측 실패: {e}")

    decision = "ALERT" if score >= BUNDLE["threshold"] else "PASS"

    # ④ 영수증 남기기
    record = log_prediction(
        application_id=req.application_id,
        score=score,
        decision=decision,
        model_version=BUNDLE["model_version"],
        threshold=BUNDLE["threshold"],
    )

    return {
        "application_id": record["application_id"],
        "score": record["score"],
        "decision": record["decision"],
        "model_version": record["model_version"],
    }