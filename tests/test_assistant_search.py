"""Hybrid search on a tiny warehouse with a deterministic fake embedder."""

import hashlib

import duckdb
import numpy as np
import pytest

from imdb_insights.assistant import search
from imdb_insights.assistant.embedder import DIMENSIONS

# Words that mean the same thing share a vector, so "vector search" can find paraphrases.
SYNONYMS = {"slow": "pace", "drags": "pace", "pacing": "pace", "sluggish": "pace"}


class FakeEmbedder:
    @staticmethod
    def _vec(text):
        v = np.zeros(DIMENSIONS, dtype=np.float32)
        for word in text.lower().replace(".", " ").split():
            word = SYNONYMS.get(word, word)
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_query(self, query):
        return self._vec(query)


PASSAGES = [
    ("r1", 1, "The story drags in the middle."),
    ("r2", 1, "Hans Zimmer delivers a thunderous score."),
    ("r3", 2, "Sluggish second act but great actors."),
    ("r4", 2, "Wonderful cinematography and score."),
]


@pytest.fixture
def con(tmp_path):
    path = tmp_path / "w.duckdb"
    with duckdb.connect(str(path)) as c:
        c.execute(
            "create table movies (tmdb_id bigint, title varchar, original_title varchar, "
            "year int, tmdb_votes int)"
        )
        c.execute(
            "insert into movies values (1, 'Interstellar', 'Interstellar', 2014, 900), "
            "(2, 'Interstellar Wars', 'Interstellar Wars', 2016, 5), "
            "(3, 'The Dark Knight', 'The Dark Knight', 2008, 800)"
        )
        c.execute(
            "create table reviews (review_id varchar, author varchar, rating double, url varchar)"
        )
        c.execute(
            f"create table review_chunks (chunk_id varchar, review_id varchar, "
            f"tmdb_id bigint, chunk_no int, text varchar, embedding FLOAT[{DIMENSIONS}])"
        )
        for rid, tmdb_id, text in PASSAGES:
            c.execute("insert into reviews values (?, 'someone', 8, 'https://x/' || ?)", [rid, rid])
            c.execute(
                "insert into review_chunks values (?, ?, ?, 0, ?, ?)",
                [f"{rid}-0", rid, tmdb_id, text, FakeEmbedder._vec(text).tolist()],
            )
        c.execute("install fts; load fts")
        c.execute("pragma create_fts_index('review_chunks', 'chunk_id', 'text')")
    connection = search.connect(path)
    yield connection
    connection.close()


def test_keyword_match_ranks_first(con):
    hits = search.hybrid_search(con, FakeEmbedder(), "Hans Zimmer", k=2)
    assert hits["review_id"].iloc[0] == "r2"
    assert hits["in_keyword"].iloc[0]


def test_paraphrase_is_found_by_vector_search(con):
    hits = search.hybrid_search(con, FakeEmbedder(), "is the pacing slow", k=2)
    assert set(hits["review_id"]) == {"r1", "r3"}  # neither contains "pacing" or "slow"
    assert not hits["in_keyword"].any() and hits["in_vector"].all()


def test_movie_filter_restricts_results(con):
    hits = search.hybrid_search(con, FakeEmbedder(), "score", tmdb_id=2)
    assert set(hits["tmdb_id"]) == {2}
    assert {"title", "url", "author", "rating"} <= set(hits.columns)


def test_find_movie_prefers_exact_title_then_votes(con):
    assert search.find_movie(con, "interstellar").tmdb_id == 1
    assert search.find_movie(con, "dark knight").title == "The Dark Knight"
    assert search.find_movie(con, "No Such Film") is None
