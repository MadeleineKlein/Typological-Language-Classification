from src.features.linguistic_features import LinguisticStatsTransformer
from src.features.text_vectorizer import build_char_vectorizer, build_word_vectorizer


def test_char_vectorizer_builds():
    vectorizer = build_char_vectorizer(ngram_range=(1, 2), max_features=100)
    X = vectorizer.fit_transform(["hello", "bonjour"])
    assert X.shape[0] == 2
    assert X.shape[1] <= 100


def test_word_vectorizer_builds():
    vectorizer = build_word_vectorizer(ngram_range=(1, 2), max_features=50)
    X = vectorizer.fit_transform(["hello world", "hola mundo"])
    assert X.shape[0] == 2
    assert X.shape[1] <= 50


def test_linguistic_stats_transformer_outputs_expected_shape():
    transformer = LinguisticStatsTransformer()
    X = transformer.transform(["Hello!", "Ça va?"])
    assert X.shape == (2, len(transformer.feature_names_))

