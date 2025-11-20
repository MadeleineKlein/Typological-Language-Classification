"""
Evaluation helpers for language identification models.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def classification_summary(y_true, y_pred) -> dict:
    payload = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    payload["accuracy"] = accuracy_score(y_true, y_pred)
    return payload


def decision_scores(pipeline: Pipeline, texts: Iterable[str]) -> np.ndarray:
    vectorizer = pipeline.named_steps["vectorizer"]
    clf = pipeline.named_steps["classifier"]
    X = vectorizer.transform(texts)
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
    elif hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X)
    else:
        raise ValueError("Classifier must expose decision_function or predict_proba.")
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    return scores


def top_k_accuracy(
    pipeline: Pipeline,
    texts: Iterable[str],
    labels: Sequence[str],
    k: int = 3,
) -> float:
    scores = decision_scores(pipeline, texts)
    classes = pipeline.named_steps["classifier"].classes_
    indices = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    predicted = classes[indices]
    total = len(labels)
    hits = sum(true in pred_row for true, pred_row in zip(labels, predicted))
    return hits / total if total else 0.0


def compute_confusion(y_true, y_pred, labels) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


__all__ = [
    "classification_summary",
    "top_k_accuracy",
    "compute_confusion",
    "decision_scores",
]

