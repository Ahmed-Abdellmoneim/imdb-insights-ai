import pandas as pd
import pytest

from imdb_insights.reviews import data


def _reviews(rows):
    return pd.DataFrame(
        [
            {
                "review_id": rid,
                "source": "stanford",
                "split": split,
                "imdb_id": movie,
                "text": text,
                "rating": rating,
            }
            for rid, split, movie, text, rating in rows
        ]
    )


def test_clean_text_strips_markup():
    assert data.clean_text("Great!<br /><br />Loved it &amp; more") == "Great!\n\nLoved it & more"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I give it 7/10.", True),
        ("solid 8 out of 10", True),
        ("a 4.5 / 10 at best", True),
        ("10/10 masterpiece", True),
        ("Released 10/10/2005 in cinemas", False),
        ("on 3/10/99 I saw it", False),
        ("one of 10 films", False),
        ("2 of 10 friends liked it", False),
        ("no score here", False),
    ],
)
def test_states_score(text, expected):
    assert data.states_score(pd.Series([text])).iloc[0] == expected


def test_validation_split_is_grouped_by_movie():
    rows = [(f"r{i}", "train", f"tt{i % 20}", f"text {i}", 1 + i % 10) for i in range(400)]
    train, validation = data.split_validation(_reviews(rows), fraction=0.25)
    assert not set(train["imdb_id"]) & set(validation["imdb_id"])
    assert len(set(validation["imdb_id"])) == 5


def test_build_splits_dedupes_train_and_flags_leaks():
    stanford = _reviews(
        [(f"tr{i}", "train", f"tt{i}", f"unique train review {i}", 8) for i in range(50)]
        + [("dup", "train", "tt0", "Unique Train Review 0", 8)]  # duplicate (case/space)
        + [
            ("t1", "test", "tt900", "a fresh test review", 3),
            ("t2", "test", "tt901", "unique train review 3", 9),  # text seen in train
        ]
    )
    splits = data.build_splits(stanford)
    assert "dup" not in set(splits["review_id"])
    flags = splits.set_index("review_id")["leaked"]
    assert not flags["t1"]
    # t2 repeats tr3's text: leaked exactly when tr3 ended up in train (not validation).
    tr3_in_train = "tr3" in set(splits.query("split == 'train'")["review_id"])
    assert flags["t2"] == tr3_in_train
    assert not splits.query("split == 'train'")["leaked"].any()
