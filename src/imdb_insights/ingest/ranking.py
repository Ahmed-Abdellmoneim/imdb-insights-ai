"""Rank movies the way IMDb's Top 250 does: a Bayesian weighted rating.

    WR = v / (v + m) * R + m / (v + m) * C

R is the movie's mean rating, v its vote count, m the minimum votes to be listed, and C the
prior mean rating. A movie with few votes is pulled towards C; one with millions of votes
keeps essentially its own rating. IMDb published m = 25,000 and C = 7.0 for this chart.
"""

from __future__ import annotations

import pandas as pd

MIN_VOTES = 25_000
PRIOR_MEAN = 7.0


def weighted_rating(
    rating: pd.Series, votes: pd.Series, m: int = MIN_VOTES, c: float = PRIOR_MEAN
) -> pd.Series:
    return votes / (votes + m) * rating + m / (votes + m) * c


def rank_top(
    movies: pd.DataFrame,
    n: int = 250,
    *,
    rating: str = "imdb_rating",
    votes: str = "imdb_votes",
    m: int = MIN_VOTES,
    c: float = PRIOR_MEAN,
) -> pd.DataFrame:
    """Return ``movies`` with ``weighted_rating`` and ``rank`` (1..n, else <NA>) columns."""
    out = movies.copy()
    eligible = out[rating].notna() & (out[votes] >= m)
    out["weighted_rating"] = weighted_rating(out[rating], out[votes], m, c).where(eligible)
    order = out.loc[eligible].sort_values(["weighted_rating", votes], ascending=False).index[:n]
    out["rank"] = pd.Series(range(1, len(order) + 1), index=order, dtype="Int64")
    return out
