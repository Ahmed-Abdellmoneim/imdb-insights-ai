"""What a submitted review is, the checks it goes through, and how approved ones are used.

Validation errors stop a submission (the form shows them). Flags do not: they point the
moderator at things worth a closer look, such as a link or a phone number.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

MIN_CHARS = 150
MAX_CHARS = 6000
MIN_WORDS = 30
MAX_HEADLINE = 100
MAX_AUTHOR = 40

CONSENT_VERSION = "2026-10-16"
CONSENT_TEXT = (
    "I wrote this review myself. Once a moderator approves it, it may be shown in this app "
    "(under the name I give, or as Anonymous), quoted by its assistant, and used to evaluate "
    "and train rating models, under the CC BY 4.0 licence."
)

REJECT_REASONS = {
    "s": "spam or advertising",
    "o": "not a review of this movie",
    "a": "abusive or hateful",
    "p": "contains personal information",
    "c": "copied from elsewhere",
    "d": "duplicate",
    "x": "other",
}

_LINK = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|net|org|io|ly|me)\b", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
_REPEATED = re.compile(r"(.)\1{7,}")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


class Submission(BaseModel):
    submission_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tmdb_id: int
    movie_title: str
    rating: int = Field(ge=1, le=10)
    headline: str | None = Field(default=None, max_length=MAX_HEADLINE)
    text: str = Field(min_length=MIN_CHARS, max_length=MAX_CHARS)
    author: str | None = Field(default=None, max_length=MAX_AUTHOR)
    consent: str = CONSENT_VERSION
    # A random id per browser session (no IP address or account), to spot floods.
    session: str
    # The model's guess, made after the writer chose their rating and shown to them only
    # afterwards, so it cannot influence the rating it is evaluated against.
    model_rating: float | None = None
    model_version: str | None = None
    flags: list[str] = Field(default_factory=list)

    @property
    def date_path(self) -> str:
        return f"submissions/{self.created_at.date().isoformat()}/{self.submission_id}.json"


class Decision(BaseModel):
    submission_id: str
    status: Literal["approved", "rejected"]
    reason: str | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def path(self) -> str:
        return f"moderation/{self.submission_id}.json"


def clean(text: str | None) -> str:
    """Trim each line and keep at most one blank line between paragraphs."""
    lines = [line.strip() for line in (text or "").replace("\r\n", "\n").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def problems(text: str, headline: str = "", author: str = "") -> list[str]:
    """Reasons the form refuses a review (empty when it is acceptable)."""
    errors = []
    if len(text) < MIN_CHARS:
        errors.append(f"Please write at least {MIN_CHARS} characters (now {len(text)}).")
    elif len(_WORD.findall(text)) < MIN_WORDS:
        errors.append(f"Please write at least {MIN_WORDS} words.")
    if len(text) > MAX_CHARS:
        errors.append(f"Please keep it under {MAX_CHARS:,} characters (now {len(text):,}).")
    if len(headline) > MAX_HEADLINE:
        errors.append(f"The headline can be at most {MAX_HEADLINE} characters.")
    if len(author) > MAX_AUTHOR:
        errors.append(f"The name can be at most {MAX_AUTHOR} characters.")
    return errors


def flags(text: str, headline: str = "", author: str = "") -> list[str]:
    """Things a moderator should look at; they never block a submission on their own."""
    everything = "\n".join([headline, author, text])
    found = []
    if _LINK.search(everything):
        found.append("link")
    if _EMAIL.search(everything):
        found.append("email address")
    if _PHONE.search(everything):
        found.append("phone number")
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        found.append("mostly capitals")
    if _REPEATED.search(text):
        found.append("repeated characters")
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) >= MIN_WORDS and len(set(words)) / len(words) < 0.3:
        found.append("repetitive wording")
    return found


def shingles(text: str, size: int = 8) -> set[tuple[str, ...]]:
    words = [w.lower() for w in _WORD.findall(text)]
    return {tuple(words[i : i + size]) for i in range(max(0, len(words) - size + 1))}


def overlap(a: set, b: set) -> float:
    """Share of ``a``'s 8-word sequences that also appear in ``b`` (copy detection)."""
    return len(a & b) / len(a) if a else 0.0


def latest(decisions: Iterable[Decision]) -> dict[str, Decision]:
    by_id: dict[str, Decision] = {}
    for d in sorted(decisions, key=lambda d: d.decided_at):
        by_id[d.submission_id] = d
    return by_id


REVIEW_COLUMNS = ["review_id", "tmdb_id", "source", "author", "rating", "content",
                  "created_at", "url"]  # fmt: skip


def approved_reviews(
    submissions: Iterable[Submission], decisions: dict[str, Decision]
) -> pd.DataFrame:
    """Approved submissions as rows of the warehouse's ``reviews`` table."""
    rows = [
        {
            "review_id": f"community-{s.submission_id}",
            "tmdb_id": s.tmdb_id,
            "source": "community",
            "author": s.author or "Anonymous",
            "rating": float(s.rating),
            "content": f"{s.headline}\n\n{s.text}" if s.headline else s.text,
            "created_at": s.created_at,
            "url": None,
        }
        for s in submissions
        if (d := decisions.get(s.submission_id)) and d.status == "approved"
    ]
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)
