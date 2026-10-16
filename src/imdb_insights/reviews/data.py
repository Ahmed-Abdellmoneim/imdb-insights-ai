"""Review datasets with leakage-safe splits.

* **Stanford Large Movie Review Dataset** (Maas et al., 2011): 50k labelled IMDb reviews,
  ratings 1-4 and 7-10 (no neutral 5-6), with an official train/test split in which no movie
  appears in both halves. We carve a validation set out of *train*, grouped by movie, so the
  test set is only touched for final numbers.
* **TMDB reviews** from our warehouse: an out-of-domain test set that, unlike Stanford,
  includes the 5-6 middle of the scale.

Every frame has the columns ``review_id, source, split, imdb_id, text, rating``.

Leakage found in the official Stanford split (measured, see ``build_splits``): 123 test
reviews also appear word-for-word in train, and one movie (tt0406816, 23 test reviews) is in
both halves. Test rows are flagged ``leaked`` so results can be reported on the official set
(comparable with the literature) and on the clean remainder (the honest number).
"""

from __future__ import annotations

import hashlib
import html
import re
import tarfile
from pathlib import Path

import duckdb
import httpx
import numpy as np
import pandas as pd

STANFORD_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
STANFORD_SHA256 = "c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe"
COLUMNS = ["review_id", "source", "split", "imdb_id", "text", "rating"]


def clean_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return re.sub(r"[ \t]+", " ", text).strip()


def download_stanford(external_dir: Path) -> Path:
    """Download and safely extract the dataset (once); returns the ``aclImdb`` folder."""
    root = external_dir / "aclImdb"
    if root.exists():
        return root
    external_dir.mkdir(parents=True, exist_ok=True)
    archive = external_dir / "aclImdb_v1.tar.gz"
    if not archive.exists():
        with httpx.stream("GET", STANFORD_URL, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with archive.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != STANFORD_SHA256:
        archive.unlink()
        raise ValueError(f"Checksum mismatch for {archive.name}: {digest}")
    with tarfile.open(archive, "r:gz") as tar:
        # The "data" filter rejects absolute paths and links that escape the target.
        tar.extractall(external_dir, filter="data")
    return root


def _imdb_id(url: str) -> str | None:
    match = re.search(r"title/(tt\d+)", url)
    return match.group(1) if match else None


def load_stanford_cached(external_dir: Path) -> pd.DataFrame:
    """``load_stanford`` with a Parquet cache (reading 50k small files is slow on Windows)."""
    cache = external_dir / "reviews.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = load_stanford(download_stanford(external_dir))
    df.to_parquet(cache, index=False)
    return df


def load_stanford(root: Path) -> pd.DataFrame:
    """Read the labelled train and test reviews (the unlabelled ``unsup`` set is skipped)."""
    rows = []
    for split in ("train", "test"):
        for polarity in ("pos", "neg"):
            folder = root / split / polarity
            # urls_<polarity>.txt line i belongs to the review file with id i.
            urls = (root / split / f"urls_{polarity}.txt").read_text(encoding="utf-8").split()
            for path in folder.glob("*.txt"):
                review_index, rating = path.stem.split("_")
                rows.append(
                    {
                        "review_id": f"stanford-{split}-{polarity}-{review_index}",
                        "source": "stanford",
                        "split": split,
                        "imdb_id": _imdb_id(urls[int(review_index)]),
                        "text": clean_text(path.read_text(encoding="utf-8")),
                        "rating": int(rating),
                    }
                )
    return pd.DataFrame(rows, columns=COLUMNS)


def load_tmdb(db_path: Path) -> pd.DataFrame:
    """TMDB reviews that carry the reviewer's own rating, as an out-of-domain test set."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(
            """
            select r.review_id, m.imdb_id, r.content as text, r.rating
            from reviews r join movies m using (tmdb_id)
            where r.source = 'tmdb' and r.rating is not null and length(r.content) >= 50
            """
        ).df()
    df["text"] = df["text"].map(clean_text)
    # TMDB allows half points; round to the 1-10 integer scale used everywhere else.
    df["rating"] = df["rating"].clip(1, 10).round().astype(int)
    df["source"] = "tmdb"
    df["split"] = "test_ood"
    df["review_id"] = "tmdb-" + df["review_id"]
    return df[COLUMNS]


def split_validation(
    train: pd.DataFrame, fraction: float = 0.1, seed: int = 13
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out ``fraction`` of the movies (not reviews), so no movie leaks across."""
    movies = train["imdb_id"].fillna(train["review_id"]).unique()
    rng = np.random.default_rng(seed)
    held_out = set(rng.choice(movies, size=max(1, int(len(movies) * fraction)), replace=False))
    is_val = train["imdb_id"].fillna(train["review_id"]).isin(held_out)
    return train[~is_val].assign(split="train"), train[is_val].assign(split="validation")


# "7/10", "7 / 10", "4.5/10", "7 out of 10": the reviewer states the answer in the text.
# Not dates such as "10/10/2005" (hence the look-arounds).
STATED_SCORE = re.compile(
    r"(?<![\d/.])(?:10|[0-9](?:\.[0-9])?)\s*(?:/|out\s+of)\s*10(?![\d/])", re.IGNORECASE
)


def states_score(text: pd.Series) -> pd.Series:
    """Whether each review spells out its own x/10 score."""
    return text.str.contains(STATED_SCORE, regex=True)


def _normalized(text: pd.Series) -> pd.Series:
    return text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def build_splits(stanford: pd.DataFrame, tmdb: pd.DataFrame | None = None) -> pd.DataFrame:
    """One frame with ``train``, ``validation``, ``test`` and (optionally) ``test_ood`` rows.

    * Duplicate texts inside Stanford train are dropped (one copy kept).
    * Validation is a movie-grouped slice of train, so tuning never sees test movies.
    * ``leaked`` marks evaluation rows whose text, or whose movie, also occurs in training.
    """
    stanford_train = stanford[stanford["split"] == "train"]
    stanford_train = stanford_train[~_normalized(stanford_train["text"]).duplicated()]
    train, validation = split_validation(stanford_train)
    parts = [train, validation, stanford[stanford["split"] == "test"]]
    if tmdb is not None and len(tmdb):
        parts.append(tmdb)
    df = pd.concat(parts, ignore_index=True)

    fit_rows = df["split"] == "train"
    seen_texts = set(_normalized(df.loc[fit_rows, "text"]))
    seen_movies = set(df.loc[fit_rows, "imdb_id"].dropna())
    df["leaked"] = ~fit_rows & (
        _normalized(df["text"]).isin(seen_texts) | df["imdb_id"].isin(seen_movies)
    )
    df["states_score"] = states_score(df["text"])
    return df
