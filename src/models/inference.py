"""
Inference helpers for the language identification model.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from src.config.settings import DEFAULT_MODEL_PATH


def load_pipeline(model_path: Path | None = None) -> Pipeline:
    """
    Load a serialized sklearn Pipeline containing the vectorizer + classifier.
    """
    path = model_path or DEFAULT_MODEL_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train a model with "
            "`python -m src.models.training_pipeline sample` first."
        )
    return joblib.load(path)


def predict_language(
    text: str,
    pipeline: Pipeline | None = None,
    model_path: Path | None = None,
) -> dict:
    """
    Run inference on an input string and return probabilities / scores.
    """
    if pipeline is None:
        pipeline = load_pipeline(model_path)
    clf = pipeline.named_steps["classifier"]
    vectorizer = pipeline.named_steps["vectorizer"]
    features = vectorizer.transform([text])

    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(features)[0]
    elif hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(features)[0]
    else:
        raise ValueError("Classifier does not expose decision_function or predict_proba.")

    classes: List[str] = list(clf.classes_)
    probs = np.exp(scores - np.max(scores))
    probs = probs / probs.sum()
    ranked = sorted(
        [
            {"language": lang, "score": float(score), "probability": float(prob)}
            for lang, score, prob in zip(classes, scores, probs)
        ],
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "top_language": ranked[0]["language"],
        "top_score": ranked[0]["score"],
        "top_probability": ranked[0]["probability"],
        "ranked": ranked,
    }


__all__ = ["load_pipeline", "predict_language"]



