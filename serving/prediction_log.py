# serving/prediction_log.py
import json
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl"))


def log_prediction(application_id, score, decision, model_version, threshold):
    """판정 1건을 jsonl 한 줄로 append. 기록한 딕셔너리를 그대로 반환."""
    record = {
        "application_id": application_id,
        "score": score,
        "decision": decision,
        "model_version": model_version,
        "threshold": threshold,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record