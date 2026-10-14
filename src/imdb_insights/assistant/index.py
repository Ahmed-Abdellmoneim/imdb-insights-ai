"""Turn reviews into searchable passages: chunking, cached embeddings, and the chunk table.

Short passages (a few sentences each) make retrieval precise and citations specific.
Embeddings are cached by a hash of the passage text, so the daily refresh only embeds
reviews it has not seen before (the cache travels with the rest of the API cache).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .embedder import DIMENSIONS, Embedder

log = logging.getLogger(__name__)

TARGET_CHARS = 700  # ~170 tokens: a few sentences
HARD_MAX_CHARS = 1200
CHUNK_COLUMNS = ["chunk_id", "review_id", "tmdb_id", "chunk_no", "text", "embedding"]


def _split_long(paragraph: str) -> list[str]:
    """Split one long paragraph at sentence ends; cut mid-sentence only as a last resort."""
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > HARD_MAX_CHARS:
            pieces.append(sentence[:HARD_MAX_CHARS])
            sentence = sentence[HARD_MAX_CHARS:]
        if current and len(current) + 1 + len(sentence) > TARGET_CHARS:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str) -> list[str]:
    """Passages of about ``TARGET_CHARS``, respecting paragraph and sentence boundaries."""
    pieces = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if paragraph:
            pieces += [paragraph] if len(paragraph) <= HARD_MAX_CHARS else _split_long(paragraph)
    chunks, current = [], ""
    for piece in pieces:  # merge small neighbouring pieces
        if current and len(current) + 1 + len(piece) > TARGET_CHARS:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Passage-text hash -> embedding, persisted as Parquet."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.vectors: dict[str, np.ndarray] = {}
        if path.exists():
            df = pd.read_parquet(path)
            self.vectors = dict(zip(df["text_hash"], df["embedding"].map(np.asarray), strict=True))

    def embed(self, embedder: Embedder, texts: list[str]) -> np.ndarray:
        hashes = [_text_hash(t) for t in texts]
        missing = list(dict.fromkeys(h for h in hashes if h not in self.vectors))
        if missing:
            by_hash = {h: t for h, t in zip(hashes, texts, strict=True)}
            log.info("Embedding %d new passages (%d cached)", len(missing), len(self.vectors))
            new = embedder.embed_passages([by_hash[h] for h in missing])
            self.vectors.update(zip(missing, new, strict=True))
        return np.stack([self.vectors[h] for h in hashes]) if hashes else np.empty((0, DIMENSIONS))

    def save(self, keep: set[str] | None = None) -> None:
        """Persist, optionally dropping entries for passages that no longer exist."""
        items = [(h, v) for h, v in self.vectors.items() if keep is None or h in keep]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"text_hash": [h for h, _ in items], "embedding": [v.tolist() for _, v in items]}
        ).to_parquet(self.path, index=False)


def build_review_chunks(
    reviews: pd.DataFrame, embedder: Embedder, cache: EmbeddingCache
) -> pd.DataFrame:
    rows = []
    for review in reviews.itertuples():
        for n, text in enumerate(chunk_text(review.content)):
            rows.append({"chunk_id": f"{review.review_id}-{n}", "review_id": review.review_id,
                         "tmdb_id": review.tmdb_id, "chunk_no": n, "text": text})  # fmt: skip
    chunks = pd.DataFrame(rows, columns=CHUNK_COLUMNS[:-1])
    vectors = cache.embed(embedder, chunks["text"].tolist())
    chunks["embedding"] = [v.astype(np.float32).tolist() for v in vectors]
    cache.save(keep={_text_hash(t) for t in chunks["text"]})
    return chunks
