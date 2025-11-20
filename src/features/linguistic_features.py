"""
Feature transformers that capture handcrafted linguistic statistics.
"""
from __future__ import annotations

import string
import unicodedata
from typing import Iterable, List

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


def _diacritic_count(text: str) -> int:
    return sum(
        1
        for char in text
        if unicodedata.category(char).startswith("L") and ord(char) > 127
    )


class LinguisticStatsTransformer(BaseEstimator, TransformerMixin):
    """
    Generate simple numeric features describing text composition.
    """

    feature_names_ = [
        "char_len",
        "whitespace_ratio",
        "punct_ratio",
        "digit_ratio",
        "uppercase_ratio",
        "diacritic_ratio",
        "vowel_ratio",
    ]

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]) -> sparse.csr_matrix:
        rows: List[List[float]] = []
        vowels = set("aeiouyàáâäãåèéêëìíîïòóôöõùúûüñýÿæœ")

        for text in X:
            text = text or ""
            length = len(text)
            whitespace = sum(1 for c in text if c.isspace())
            punctuation = sum(1 for c in text if c in string.punctuation)
            digits = sum(1 for c in text if c.isdigit())
            uppercase = sum(1 for c in text if c.isupper())
            diacritics = _diacritic_count(text)
            letters = sum(1 for c in text if c.isalpha())
            vowel_count = sum(1 for c in text.lower() if c in vowels)

            safe_len = max(length, 1)
            safe_letters = max(letters, 1)
            rows.append(
                [
                    float(length),
                    whitespace / safe_len,
                    punctuation / safe_len,
                    digits / safe_len,
                    uppercase / safe_len,
                    diacritics / safe_len,
                    vowel_count / safe_letters,
                ]
            )
        return sparse.csr_matrix(np.asarray(rows, dtype=float))


__all__ = ["LinguisticStatsTransformer"]

