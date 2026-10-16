"""The rating model the app runs on submitted reviews: bge-small embeddings + a ridge head.

Step 3's embedding model, minus PyTorch: the encoder is the ONNX export the assistant
already uses, and the head (384 weights and an intercept) ships with the code as JSON. It
is trained by ``python -m imdb_insights.reviews train embeddings``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..assistant.embedder import DIMENSIONS, Embedder

HEAD_PATH = Path(__file__).with_name("rating_head.json")


def save_head(head: dict, path: Path = HEAD_PATH) -> Path:
    if len(head["coef"]) != DIMENSIONS:
        raise ValueError(f"Expected {DIMENSIONS} weights, got {len(head['coef'])}")
    path.write_text(json.dumps(head, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


@dataclass(frozen=True)
class RatingHead:
    coef: np.ndarray
    intercept: float
    version: str  # hash of the head's contents, recorded with every prediction

    @classmethod
    def load(cls, path: Path = HEAD_PATH) -> RatingHead:
        head = json.loads(path.read_text(encoding="utf-8"))
        # Hash the parsed numbers, not the bytes, so line endings cannot change the version.
        canonical = json.dumps(head, sort_keys=True, separators=(",", ":")).encode()
        return cls(
            coef=np.asarray(head["coef"], dtype=np.float32),
            intercept=float(head["intercept"]),
            version=hashlib.sha256(canonical).hexdigest()[:10],
        )

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Ratings on the 1-10 scale for L2-normalized passage embeddings."""
        return np.clip(embeddings @ self.coef + self.intercept, 1.0, 10.0)


class ReviewRater:
    def __init__(self, embedder: Embedder, head: RatingHead | None = None) -> None:
        self.embedder = embedder
        self.head = head or RatingHead.load()

    @property
    def version(self) -> str:
        return self.head.version

    def rate(self, text: str) -> float:
        return round(float(self.head.predict(self.embedder.embed_passages([text]))[0]), 2)
