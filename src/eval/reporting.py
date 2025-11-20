"""
Persist evaluation artifacts such as JSON metrics and confusion matrices.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


def save_metrics_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_confusion_plot(
    matrix,
    labels: List[str],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Language Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


__all__ = ["save_metrics_json", "save_confusion_plot"]

