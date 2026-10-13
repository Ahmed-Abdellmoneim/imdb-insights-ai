"""Rating models. Each one fits on ``train``, tunes on ``validation`` and predicts 1-10.

Models live in separate modules so heavy imports (torch, transformers) only happen for the
model actually being trained.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class RatingModel(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> dict:
        """Train; return what was learned or chosen (hyperparameters, validation scores)."""

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predicted ratings on the 1-10 scale (not necessarily integers)."""


def get_model(name: str) -> RatingModel:
    if name == "tfidf":
        from .tfidf import TfidfRidge

        return TfidfRidge()
    if name == "embeddings":
        from .embeddings import EmbeddingRidge

        return EmbeddingRidge()
    if name == "finetune":
        from .finetune import FineTunedRegressor

        return FineTunedRegressor()
    raise ValueError(f"Unknown model {name!r}; choose tfidf, embeddings or finetune")
