"""
Utilities for building TF-IDF vectorizers targeting language cues.
"""
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def build_char_vectorizer(
    ngram_range: Tuple[int, int] = (1, 3),
    max_features: int | None = 40_000,
    use_idf: bool = True,
) -> TfidfVectorizer:
    """
    Create a char-level TF-IDF vectorizer tailored for language identification.
    """
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features,
        use_idf=use_idf,
        lowercase=True,
        norm="l2",
        sublinear_tf=True,
    )


def build_word_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int | None = 20_000,
    use_idf: bool = True,
) -> TfidfVectorizer:
    """
    Capture lexical cues via word n-grams.
    """
    return TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=ngram_range,
        max_features=max_features,
        use_idf=use_idf,
        lowercase=True,
        norm="l2",
        sublinear_tf=True,
    )


__all__ = ["build_char_vectorizer", "build_word_vectorizer"]



