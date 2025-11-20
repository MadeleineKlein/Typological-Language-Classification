import pandas as pd

from src.config.settings import SAMPLE_DATA, TrainingConfig
from src.models.inference import predict_language
from src.models.training_pipeline import train_and_eval


def _train_sample_pipeline():
    df = pd.read_csv(SAMPLE_DATA)
    config = TrainingConfig()
    pipeline, _, _ = train_and_eval(df, config, grid_search=False)
    return pipeline


def test_predict_language_returns_ranked_probabilities():
    pipeline = _train_sample_pipeline()
    text = "Hello, we are testing the classifier."
    result = predict_language(text, pipeline=pipeline)
    assert "ranked" in result
    assert result["ranked"][0]["probability"] <= 1
    assert len(result["ranked"]) >= 1
    assert result["top_language"] in [entry["language"] for entry in result["ranked"]]

