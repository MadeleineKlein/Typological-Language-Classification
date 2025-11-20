"""
Global configuration for the typological language identification project.
"""
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA = DATA_DIR / "sample" / "sample_sentences.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

LANGUAGE_CONFIG_PATH = Path(__file__).resolve().parent / "languages.yaml"
with LANGUAGE_CONFIG_PATH.open("r", encoding="utf-8") as f:
    _lang_payload = yaml.safe_load(f) or {}
_language_entries = _lang_payload.get("languages", [])
TARGET_LANGUAGES: List[str] = [entry["code"] for entry in _language_entries]
LANGUAGE_METADATA: Dict[str, dict] = {
    entry["code"]: entry for entry in _language_entries
}

# Path to the serialized sklearn Pipeline used for inference.
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "sample_language_svm.joblib"


class TrainingConfig:
    """Hyperparameters used for SVM training."""

    char_ngram_range = (1, 4)
    char_max_features = 60_000
    word_ngram_range = (1, 2)
    word_max_features = 20_000
    use_idf = True
    include_word_features = True
    include_stats_features = True
    c_value = 1.0
    class_weight = "balanced"
    test_size = 0.2
    random_state = 42
    top_k = 3

    param_grid = {
        "classifier__C": [0.5, 1.0, 2.0],
        "vectorizer__char__ngram_range": [(1, 3), (1, 4)],
        "vectorizer__word__ngram_range": [(1, 1), (1, 2)],
    }


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "SAMPLE_DATA",
    "ARTIFACTS_DIR",
    "LANGUAGE_CONFIG_PATH",
    "TARGET_LANGUAGES",
    "LANGUAGE_METADATA",
    "DEFAULT_MODEL_PATH",
    "TrainingConfig",
]

