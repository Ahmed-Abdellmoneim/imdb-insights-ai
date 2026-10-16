"""The "Write a review" page: first-party reviews, read by a moderator before they are shown."""

from __future__ import annotations

import logging
import time
import uuid

import streamlit as st

from ..community.submissions import (
    CONSENT_TEXT,
    MAX_AUTHOR,
    MAX_CHARS,
    MAX_HEADLINE,
    MIN_CHARS,
    Submission,
    clean,
    flags,
    problems,
)
from ..config import get_settings
from . import data
from .assistant_page import _embedder

log = logging.getLogger(__name__)

SUBMISSIONS_PER_HOUR = 3  # per browser session; moderation catches the rest


@st.cache_resource
def _store():
    from ..community.cli import open_store

    return open_store()


@st.cache_resource(show_spinner="Loading the rating model…")
def _rater():
    from ..community.rater import ReviewRater

    return ReviewRater(_embedder())


def collection_enabled() -> bool:
    """Deployed apps need the community dataset; local runs save to ``data/community``."""
    settings = get_settings()
    return bool(settings.hf_community_repo) or not settings.hf_dataset_repo


def _guess(text: str) -> tuple[float | None, str | None]:
    try:
        rater = _rater()
        return rater.rate(text), rater.version
    except Exception:  # noqa: BLE001 - the guess is a bonus; never lose a review over it
        log.exception("Rating model unavailable")
        return None, None


def _show_result(result: dict) -> None:
    st.success(
        f"Thank you! Your review of **{result['title']}** is saved and will appear once a "
        "moderator has read it (usually within a day).",
        icon=":material/check_circle:",
    )
    if result["guess"] is None:
        return
    left, right = st.columns(2)
    left.metric("Your rating", f"{result['rating']}/10")
    right.metric(
        "Our model's guess",
        f"{result['guess']:.1f}/10",
        delta=f"{result['guess'] - result['rating']:+.1f}",
        delta_color="off",
    )
    st.caption(
        "The model read only your text. It was trained on 25,000 IMDb reviews from the Stanford "
        "Large Movie Review Dataset; approved reviews like yours measure how well it does on "
        "reviews it has never seen. It guessed after you chose your rating, so it could not "
        "influence it."
    )


def review_page() -> None:
    st.title("Write a review")
    st.caption(
        "Reviews written here are read by a person before they appear on the movie's page and "
        "in the assistant's answers. Please don't include personal details."
    )
    state = st.session_state
    state.setdefault("session_id", uuid.uuid4().hex[:12])
    state.setdefault("submitted", [])  # (time, text) pairs for this browser session

    if "review_result" in state:
        _show_result(state.pop("review_result"))
    if not collection_enabled():
        st.info("Review collection is not switched on in this deployment yet.")
        return

    movies = data.top250()
    labels = {row.tmdb_id: f"{row.title} ({row.year})" for row in movies.itertuples()}
    ids = sorted(labels, key=lambda i: labels[i].casefold())
    default = st.query_params.get("id")
    index = (
        ids.index(int(default))
        if default and default.isdigit() and int(default) in labels
        else None
    )

    n = state.setdefault("form_no", 0)  # a new number gives a fresh, empty form
    with st.form(f"review-{n}"):
        tmdb_id = st.selectbox(
            "Movie",
            ids,
            index=index,
            format_func=labels.get,
            placeholder="Choose a movie",
            key=f"movie-{n}",
        )
        rating = st.pills(
            "Your rating (1–10)", list(range(1, 11)), selection_mode="single", key=f"rating-{n}"
        )
        headline = st.text_input("Headline (optional)", max_chars=MAX_HEADLINE, key=f"headline-{n}")
        text = st.text_area(
            "Your review",
            height=260,
            max_chars=MAX_CHARS,
            placeholder=f"What worked, what didn't, and why? At least {MIN_CHARS} characters.",
            key=f"text-{n}",
        )
        author = st.text_input(
            "Name to show (optional)",
            max_chars=MAX_AUTHOR,
            placeholder="Anonymous",
            key=f"author-{n}",
        )
        consent = st.checkbox(CONSENT_TEXT, key=f"consent-{n}")
        sent = st.form_submit_button("Submit review", type="primary")

    if not sent:
        return
    text, headline, author = clean(text), clean(headline), clean(author)
    errors = problems(text, headline, author)
    if tmdb_id is None:
        errors.insert(0, "Choose a movie.")
    if rating is None:
        errors.insert(0, "Choose your rating.")
    if not consent:
        errors.append("Please tick the consent box.")
    recent = [(t, s) for t, s in state.submitted if time.time() - t < 3600]
    if any(s == text for _, s in recent):
        errors = ["You have already submitted this review."]
    elif len(recent) >= SUBMISSIONS_PER_HOUR:
        errors = [f"You can submit up to {SUBMISSIONS_PER_HOUR} reviews an hour; thank you!"]
    if errors:
        for error in errors:
            st.error(error)
        return

    with st.spinner("Saving…"):
        guess, version = _guess(text)
        submission = Submission(
            tmdb_id=tmdb_id,
            movie_title=labels[tmdb_id],
            rating=rating,
            headline=headline or None,
            text=text,
            author=author or None,
            session=state.session_id,
            model_rating=guess,
            model_version=version,
            flags=flags(text, headline, author),
        )
        try:
            _store().add_submission(submission)
        except Exception:  # noqa: BLE001 - tell the writer instead of crashing the page
            log.exception("Could not save a submission")
            st.error("Sorry, the review could not be saved right now. Please try again later.")
            return
    state.submitted = [*recent, (time.time(), text)]
    state.form_no += 1
    state.review_result = {"title": labels[tmdb_id], "rating": rating, "guess": guess}
    st.rerun()  # fresh, empty form with the result on top
