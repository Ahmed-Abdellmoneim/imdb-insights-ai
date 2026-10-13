"""Sentence embeddings (frozen encoder) with a ridge regression head."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge

from ..metrics import point_metrics

ENCODER = "BAAI/bge-small-en-v1.5"
ALPHAS = (0.1, 1.0, 10.0, 100.0)


class EmbeddingRidge:
    name = "embeddings"

    def __init__(self, encoder: str = ENCODER, batch_size: int = 64) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(encoder, device=device)
        self.encoder.max_seq_length = 512  # longer reviews are truncated (head only)
        self.batch_size = batch_size
        self.model: Ridge | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> dict:
        x_train, x_val = self.embed(train["text"]), self.embed(validation["text"])
        scores = {}
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha).fit(x_train, train["rating"])
            scores[alpha] = point_metrics(validation["rating"].to_numpy(), model.predict(x_val))
        best = min(scores, key=lambda a: scores[a]["mae"])
        self.model = Ridge(alpha=best).fit(x_train, train["rating"])
        return {
            "encoder": ENCODER,
            "alpha": best,
            "dimensions": int(x_train.shape[1]),
            "validation_mae_by_alpha": {a: round(s["mae"], 4) for a, s in scores.items()},
        }

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.model.predict(self.embed(texts))
