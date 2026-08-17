# tests/test_integration.py
import json
import os
from pathlib import Path


def test_request_flows_to_log(api_client):
    """요청 1건 → 판정 → 로그 1줄까지 끝까지 흐르는지"""
    res = api_client.post("/predict", json={
        "application_id": "test-001",
        "features": {
            "income": 0.5,
            "credit_risk_score": 120.0,
            "device_os": "linux",          # 문자열로 보냄
        },
    })

    assert res.status_code == 200
    body = res.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in {"ALERT", "PASS"}

    # 로그가 실제로 쌓였는가
    log_path = Path(os.environ["PREDICTION_LOG_PATH"])
    assert log_path.exists()

    record = json.loads(log_path.read_text(encoding="utf-8").strip().split("\n")[-1])
    assert set(record) == {
        "application_id", "score", "decision",
        "model_version", "threshold", "scored_at",
    }
    assert record["application_id"] == "test-001"
    assert record["threshold"] == 0.5        # 레지스트리에서 온 값인가