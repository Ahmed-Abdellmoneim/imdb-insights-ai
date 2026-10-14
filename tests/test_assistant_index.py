import numpy as np
import pandas as pd

from imdb_insights.assistant import index
from imdb_insights.assistant.index import HARD_MAX_CHARS, TARGET_CHARS, chunk_text


def test_short_review_is_one_chunk():
    assert chunk_text("Great movie.\n\nLoved it.") == ["Great movie.\nLoved it."]


def test_long_review_splits_at_sentences_within_limits():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= HARD_MAX_CHARS for c in chunks)
    assert all(c.endswith(".") for c in chunks)  # no sentence cut in half
    assert " ".join(chunks).split() == text.split()  # nothing lost or duplicated


def test_paragraphs_are_kept_together_when_small():
    paragraphs = ["A" * 300, "B" * 300, "C" * 300]
    chunks = chunk_text("\n\n".join(paragraphs))
    assert chunks == ["A" * 300 + "\n" + "B" * 300, "C" * 300]
    assert all(len(c) <= TARGET_CHARS for c in chunks)


def test_unbroken_text_is_hard_cut():
    chunks = chunk_text("x" * (HARD_MAX_CHARS * 2 + 10))
    assert [len(c) for c in chunks] == [HARD_MAX_CHARS, HARD_MAX_CHARS, 10]


def test_blank_text_has_no_chunks():
    assert chunk_text("  \n\n ") == []


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_passages(self, texts):
        self.calls.append(list(texts))
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def test_cache_only_embeds_new_passages_and_prunes_old_ones(tmp_path):
    reviews = pd.DataFrame(
        {"review_id": ["r1", "r2"], "tmdb_id": [1, 2], "content": ["Good film.", "Bad film."]}
    )
    cache_path = tmp_path / "cache.parquet"
    embedder = FakeEmbedder()
    chunks = index.build_review_chunks(reviews, embedder, index.EmbeddingCache(cache_path))
    assert chunks["chunk_id"].tolist() == ["r1-0", "r2-0"]
    assert embedder.calls == [["Good film.", "Bad film."]]

    # Next day: one review gone, one new; only the new text is embedded.
    reviews = pd.DataFrame(
        {"review_id": ["r2", "r3"], "tmdb_id": [2, 3], "content": ["Bad film.", "Okay film."]}
    )
    embedder.calls.clear()
    chunks = index.build_review_chunks(reviews, embedder, index.EmbeddingCache(cache_path))
    assert embedder.calls == [["Okay film."]]
    assert len(pd.read_parquet(cache_path)) == 2  # "Good film." was pruned
    assert chunks["embedding"].map(len).eq(2).all()
