# src/evaluate.py
import mlflow
import numpy as np
from sklearn.metrics import roc_curve, average_precision_score


def tpr_at_fpr(y_true, y_score, target_fpr=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return tpr[idx], thresholds[idx]


def evaluate(model, X, y):
    y_score = model.predict_proba(X)[:, 1]

    tpr5, threshold = tpr_at_fpr(y, y_score, target_fpr=0.05)
    auprc = average_precision_score(y, y_score)

    mlflow.log_metric("tpr_at_fpr5", tpr5)
    mlflow.log_metric("threshold_at_fpr5pct", threshold)   # ← 새로 추가
    mlflow.log_metric("auprc", auprc)

    print(f"TPR@FPR5%: {tpr5:.4f}")
    print(f"threshold: {threshold:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    return tpr5, threshold, auprc