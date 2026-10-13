"""Baseline: TF-IDF word and bigram features with ridge regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from ..metrics import point_metrics

ALPHAS = (0.03, 0.1, 0.3, 1.0, 3.0)


class TfidfRidge:
    name = "tfidf"

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True, max_features=300_000
        )
        self.model: Ridge | None = None

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> dict:
        x_train = self.vectorizer.fit_transform(train["text"])
        x_val = self.vectorizer.transform(validation["text"])
        scores = {}
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha).fit(x_train, train["rating"])
            scores[alpha] = point_metrics(validation["rating"].to_numpy(), model.predict(x_val))
        best = min(scores, key=lambda a: scores[a]["mae"])
        self.model = Ridge(alpha=best).fit(x_train, train["rating"])
        return {
            "alpha": best,
            "features": len(self.vectorizer.vocabulary_),
            "validation_mae_by_alpha": {a: round(s["mae"], 4) for a, s in scores.items()},
        }

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.model.predict(self.vectorizer.transform(texts))

    def top_terms(self, n: int = 15) -> dict[str, list[str]]:
        """Most positive and negative n-grams: a sanity check on what the model learned."""
        vocab = np.array(self.vectorizer.get_feature_names_out())
        order = np.argsort(self.model.coef_)
        return {
            "lowers_rating": vocab[order[:n]].tolist(),
            "raises_rating": vocab[order[-n:]][::-1].tolist(),
        }
