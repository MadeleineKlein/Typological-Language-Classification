"""
Training pipeline for the SVM language identification model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import pandas as pd
import typer
from rich import print as rprint
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config.settings import (
    DEFAULT_MODEL_PATH,
    SAMPLE_DATA,
    TARGET_LANGUAGES,
    TrainingConfig,
)
from src.eval.metrics import (
    classification_summary,
    compute_confusion,
    top_k_accuracy,
)
from src.eval.reporting import save_confusion_plot, save_metrics_json
from src.features.linguistic_features import LinguisticStatsTransformer
from src.features.text_vectorizer import (
    build_char_vectorizer,
    build_word_vectorizer,
)


def _flag_to_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "on"}
    return bool(value)


app = typer.Typer(help="Train and evaluate language identification models.")


def load_dataset(
    csv_path: Path,
    languages: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load a CSV file containing `text` and `language` columns.
    """
    df = pd.read_csv(csv_path)
    required_cols = {"text", "language"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must include columns: {required_cols}")

    if languages:
        df = df[df["language"].isin(languages)]

    if df.empty:
        raise ValueError("Dataset is empty after filtering; check language list.")
    return df


def _build_vectorizer(config: TrainingConfig) -> FeatureUnion:
    transformers = [
        (
            "char",
            build_char_vectorizer(
                ngram_range=config.char_ngram_range,
                max_features=config.char_max_features,
                use_idf=config.use_idf,
            ),
        )
    ]
    if config.include_word_features:
        transformers.append(
            (
                "word",
                build_word_vectorizer(
                    ngram_range=config.word_ngram_range,
                    max_features=config.word_max_features,
                    use_idf=config.use_idf,
                ),
            )
        )
    if config.include_stats_features:
        transformers.append(
            (
                "stats",
                Pipeline(
                    [
                        ("stats", LinguisticStatsTransformer()),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
            )
        )
    return FeatureUnion(transformers)


def build_pipeline(config: TrainingConfig) -> Pipeline:
    """
    Construct an sklearn Pipeline of vectorizer + LinearSVC.
    """
    vectorizer = _build_vectorizer(config)
    classifier = LinearSVC(
        C=config.c_value,
        class_weight=config.class_weight,
        random_state=config.random_state,
        dual="auto",
    )
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def _compute_test_size(df: pd.DataFrame, config: TrainingConfig) -> float:
    """
    Ensure the test split is large enough to contain at least one sample per class.
    """
    n_samples = len(df)
    n_classes = df["language"].nunique()
    min_fraction = n_classes / n_samples
    return max(config.test_size, min_fraction + 0.01)


def train_and_eval(
    df: pd.DataFrame,
    config: TrainingConfig,
    grid_search: bool = False,
    n_jobs: int = 1,
) -> tuple[Pipeline, dict, dict]:
    """
    Train the pipeline and return fitted model along with metrics.
    """
    test_size = _compute_test_size(df, config)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["language"],
        test_size=test_size,
        random_state=config.random_state,
        stratify=df["language"],
    )
    pipeline = build_pipeline(config)
    grid_details: dict | None = None

    if grid_search:
        search = GridSearchCV(
            pipeline,
            param_grid=config.param_grid,
            cv=3,
            n_jobs=n_jobs,
            verbose=1,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        grid_details = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
        }
    else:
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    summary = classification_summary(y_test, y_pred)
    topk = top_k_accuracy(pipeline, X_test, y_test, k=config.top_k)
    conf = compute_confusion(y_test, y_pred, labels=sorted(df["language"].unique()))
    metrics = {
        "summary": summary,
        "top_k": {"k": config.top_k, "accuracy": topk},
        "grid_search": grid_details,
    }
    return pipeline, metrics, {"confusion": conf, "y_true": y_test, "y_pred": y_pred}


def _persist_evaluation_artifacts(
    output_path: Path,
    metrics: dict,
    confusion_payload: dict,
) -> Tuple[Path, Path]:
    metrics_path = output_path.with_suffix(".metrics.json")
    conf_path = output_path.with_suffix(".confusion.png")

    save_metrics_json(metrics_path, metrics)
    save_confusion_plot(
        confusion_payload["confusion"],
        labels=sorted(set(confusion_payload["y_true"])),
        path=conf_path,
    )
    return metrics_path, conf_path


@app.command()
def sample(
    output_path: Path = typer.Option(
        DEFAULT_MODEL_PATH, help="Where to store the trained model artifact."
    ),
    dataset_path: Path = typer.Option(SAMPLE_DATA, help="CSV dataset to train on."),
    grid_search: bool = typer.Option(
        False,
        "--grid-search",
        "-g",
        help="Enable small hyperparameter grid.",
        is_flag=True,
    ),
    jobs: int = typer.Option(
        1,
        "--jobs",
        "-j",
        help="Parallel jobs for grid search.",
    ),
) -> None:
    """
    Train using the bundled miniature dataset for quick experiments.
    """
    config = TrainingConfig()
    df = load_dataset(dataset_path, TARGET_LANGUAGES)
    pipeline, metrics, confusion = train_and_eval(
        df, config, grid_search=_flag_to_bool(grid_search), n_jobs=jobs
    )
    joblib.dump(pipeline, output_path)
    metrics_path, conf_path = _persist_evaluation_artifacts(output_path, metrics, confusion)
    rprint(f"[bold green]Saved model to {output_path}")
    rprint(f"[cyan]Metrics JSON → {metrics_path}")
    rprint(f"[cyan]Confusion matrix → {conf_path}")
    rprint("Top-k accuracy:", metrics["top_k"])


@app.command()
def custom(
    dataset_path: Path = typer.Argument(..., help="CSV with text/language columns."),
    output_path: Path = typer.Option(
        DEFAULT_MODEL_PATH, help="Where to store the trained model artifact."
    ),
    languages: Optional[List[str]] = typer.Option(
        None, help="Subset of language codes to keep."
    ),
    grid_search: bool = typer.Option(
        False,
        "--grid-search",
        "-g",
        help="Enable small hyperparameter grid.",
        is_flag=True,
    ),
    jobs: int = typer.Option(
        1,
        "--jobs",
        "-j",
        help="Parallel jobs for grid search.",
    ),
) -> None:
    """
    Train on an arbitrary dataset (e.g., filtered WiLI export).
    """
    config = TrainingConfig()
    df = load_dataset(dataset_path, languages)
    pipeline, metrics, confusion = train_and_eval(
        df, config, grid_search=_flag_to_bool(grid_search), n_jobs=jobs
    )
    joblib.dump(pipeline, output_path)
    metrics_path, conf_path = _persist_evaluation_artifacts(output_path, metrics, confusion)
    rprint(f"[bold green]Saved model to {output_path}")
    rprint(f"[cyan]Metrics JSON → {metrics_path}")
    rprint(f"[cyan]Confusion matrix → {conf_path}")
    rprint("Top-k accuracy:", metrics["top_k"])


if __name__ == "__main__":
    app()


