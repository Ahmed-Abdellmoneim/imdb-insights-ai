import pandas as pd
import pytest

from imdb_insights.ingest.ranking import rank_top, weighted_rating


def test_weighted_rating_shrinks_low_vote_counts_towards_the_prior():
    rating = pd.Series([9.0, 9.0])
    votes = pd.Series([25_000, 2_500_000])
    wr = weighted_rating(rating, votes)
    assert wr[0] == pytest.approx(8.0)  # v == m: halfway between 9.0 and C = 7.0
    assert wr[1] == pytest.approx(8.98, abs=0.01)


def test_rank_top_excludes_ineligible_and_breaks_ties_by_votes():
    movies = pd.DataFrame(
        {
            "title": ["few votes", "tie low", "tie high", "best", "no rating"],
            "imdb_rating": [9.9, 8.0, 8.0, 9.0, None],
            "imdb_votes": [1_000, 100_000, 100_000, 1_000_000, 500_000],
        }
    )
    movies.loc[2, "imdb_votes"] = 100_001  # same rating, slightly more votes ranks higher
    ranked = rank_top(movies, n=2).set_index("title")
    assert ranked.loc["best", "rank"] == 1
    assert ranked.loc["tie high", "rank"] == 2
    assert pd.isna(ranked.loc["tie low", "rank"])  # beyond n
    assert pd.isna(ranked.loc["few votes", "weighted_rating"])  # under 25,000 votes
    assert pd.isna(ranked.loc["no rating", "rank"])
