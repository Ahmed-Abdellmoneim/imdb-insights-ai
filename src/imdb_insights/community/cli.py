"""``uv run python -m imdb_insights.community {init,status,moderate,evaluate}``.

Moderation is deliberately manual: nothing written in the app is shown until a person has
read it. The automatic checks only decide what the moderator is told to look at.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Annotated

import duckdb
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel

from ..config import get_settings
from ..ingest.pipeline import DB_FILENAME
from ..reviews import metrics
from . import store as stores
from .submissions import (
    CONSENT_TEXT,
    REJECT_REASONS,
    Decision,
    Submission,
    latest,
    overlap,
    shingles,
)

app = typer.Typer(add_completion=False, help="Community reviews: collection and moderation.")
console = Console()

REPORT = Path("reports") / "community_reviews.md"
COPY_THRESHOLD = 0.3  # share of 8-word sequences found in another review


def open_store() -> stores.Store:
    s = get_settings()
    token = s.hf_community_token.get_secret_value() if s.hf_community_token else None
    return stores.open_store(s.data_dir, s.hf_community_repo, token)


@app.callback()
def _setup() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()


@app.command()
def init() -> None:
    """Create the private Hugging Face dataset for submissions (safe to re-run)."""
    s = get_settings()
    if not s.hf_community_repo or not s.hf_community_token:
        console.print("Set HF_COMMUNITY_REPO and a write token (HF_COMMUNITY_TOKEN or HF_TOKEN).")
        raise typer.Exit(code=2)
    stores.create_repo(
        s.hf_community_repo, s.hf_community_token.get_secret_value(), consent=CONSENT_TEXT
    )
    console.print(f"Ready: https://huggingface.co/datasets/{s.hf_community_repo} (private)")


def _status_of(submissions: list[Submission], decisions: dict[str, Decision]) -> Counter:
    return Counter(
        decisions[s.submission_id].status if s.submission_id in decisions else "pending"
        for s in submissions
    )


@app.command()
def status() -> None:
    """How many submissions are pending, approved and rejected."""
    submissions, decisions = open_store().load()
    counts = _status_of(submissions, latest(decisions))
    console.print(
        f"{len(submissions)} submissions: {counts['pending']} pending, "
        f"{counts['approved']} approved, {counts['rejected']} rejected"
    )


def _existing_reviews() -> pd.DataFrame:
    """Reviews already in the warehouse (to catch text copied from TMDB)."""
    db = get_settings().data_dir / "processed" / DB_FILENAME
    if not db.exists():
        return pd.DataFrame(columns=["tmdb_id", "author", "content"])
    with duckdb.connect(str(db), read_only=True) as con:
        return con.execute(
            "select tmdb_id, author, content from reviews where source <> 'community'"
        ).df()


def similar_reviews(
    submission: Submission, others: list[Submission], existing: pd.DataFrame
) -> list[str]:
    """Earlier reviews that share much of this one's wording."""
    mine = shingles(submission.text)
    found = []
    for other in others:
        if other.submission_id != submission.submission_id and other.created_at < (
            submission.created_at
        ):
            score = overlap(mine, shingles(other.text))
            if score >= COPY_THRESHOLD:
                found.append(f"{score:.0%} like submission {other.submission_id[:8]}")
    for row in existing[existing["tmdb_id"] == submission.tmdb_id].itertuples():
        score = overlap(mine, shingles(row.content))
        if score >= COPY_THRESHOLD:
            found.append(f"{score:.0%} like the TMDB review by {row.author}")
    return found


def _show(submission: Submission, similar: list[str], position: str) -> None:
    guess = (
        f" · model guessed {submission.model_rating:.1f}"
        if submission.model_rating is not None
        else ""
    )
    warnings = [*submission.flags, *similar]
    body = (
        (f"[bold]{submission.headline}[/bold]\n\n" if submission.headline else "")
        + submission.text
        + (f"\n\n[yellow]Check: {'; '.join(warnings)}[/yellow]" if warnings else "")
    )
    console.print(
        Panel(
            body,
            title=f"{position} · {submission.movie_title} · {submission.rating}/10{guess}",
            subtitle=(
                f"{submission.author or 'Anonymous'} · {submission.created_at:%Y-%m-%d %H:%M} "
                f"UTC · session {submission.session[:8]}"
            ),
        )
    )


@app.command()
def moderate() -> None:
    """Read each pending submission and approve or reject it."""
    store = open_store()
    submissions, decisions = store.load()
    decided = latest(decisions)
    pending = [s for s in submissions if s.submission_id not in decided]
    if not pending:
        console.print("Nothing to moderate.")
        return
    existing = _existing_reviews()
    reasons = ", ".join(f"{k}={v}" for k, v in REJECT_REASONS.items())
    new: list[Decision] = []
    try:
        for i, submission in enumerate(pending, start=1):
            _show(submission, similar_reviews(submission, submissions, existing),
                  f"{i}/{len(pending)}")  # fmt: skip
            choice = typer.prompt("[a]pprove, [r]eject, [s]kip, [q]uit", default="s").lower()
            if choice.startswith("q"):
                break
            if choice.startswith("a"):
                new.append(Decision(submission_id=submission.submission_id, status="approved"))
            elif choice.startswith("r"):
                key = typer.prompt(f"Reason ({reasons})", default="x").lower()[:1]
                reason = REJECT_REASONS.get(key, REJECT_REASONS["x"])
                new.append(
                    Decision(
                        submission_id=submission.submission_id, status="rejected", reason=reason
                    )
                )
    finally:  # save what was decided, even after Ctrl+C
        store.add_decisions(new)
        approved = sum(d.status == "approved" for d in new)
        console.print(f"Saved {len(new)} decisions ({approved} approved).")
    if new:
        console.print("Approved reviews appear in the app after the next data refresh.")


def report(submissions: list[Submission], decisions: dict[str, Decision]) -> str:
    counts = _status_of(submissions, decisions)
    lines = [
        "# Community reviews",
        "",
        "Reviews written in the app and moderated by hand. The model's guess is made from the "
        "text alone, after the writer chose a rating, and is only shown to them afterwards.",
        "",
        f"{len(submissions)} submissions: {counts['approved']} approved, "
        f"{counts['rejected']} rejected, {counts['pending']} pending.",
        "",
    ]
    approved = [
        s
        for s in submissions
        if (d := decisions.get(s.submission_id))
        and d.status == "approved"
        and s.model_rating is not None
    ]
    if approved:
        lines += [
            "## Rating model on first-party reviews",
            "",
            "Step 3's embedding model (trained on Stanford IMDb reviews) against the rating each "
            "writer gave. 95% bootstrap intervals in brackets.",
            "",
            "| model version | reviews | MAE ↓ | within ±1 ↑ | polarity acc. ↑ | Spearman ↑ "
            "| bias |",
            "|---|---|---|---|---|---|---|",
        ]
        frame = pd.DataFrame(
            [(s.model_version, s.rating, s.model_rating) for s in approved],
            columns=["version", "rating", "prediction"],
        )
        for version, group in frame.groupby("version"):
            m = metrics.evaluate(group["rating"].to_numpy(), group["prediction"].to_numpy())
            lines.append(
                f"| `{version}` | {m['n']} | {m['mae']:.2f} [{m['mae_lo']:.2f}–{m['mae_hi']:.2f}] "
                f"| {m['within_1']:.0%} | {m['polarity_accuracy']:.0%} | {m['spearman']:.2f} "
                f"| {m['bias']:+.2f} |".replace("nan%", "–").replace("| nan |", "| – |")
            )
        lines.append("")
        stars = frame["rating"].value_counts().reindex(range(1, 11), fill_value=0)
        lines += [
            "Ratings given: " + ", ".join(f"{k}★ {v}" for k, v in stars.items() if v),
            "",
        ]
    flagged = [s for s in submissions if s.flags and s.submission_id in decisions]
    if flagged:
        rejected = sum(decisions[s.submission_id].status == "rejected" for s in flagged)
        lines += [
            "## Moderation",
            "",
            f"Automatic flags were raised on {len(flagged)} moderated submissions; "
            f"{rejected} of them were rejected ({rejected / len(flagged):.0%}).",
            "",
        ]
    reasons = Counter(d.reason for d in decisions.values() if d.status == "rejected")
    if reasons:
        lines += ["| rejected because | submissions |", "|---|---|"]
        lines += [f"| {reason} | {n} |" for reason, n in reasons.most_common()]
        lines.append("")
    return "\n".join(lines)


@app.command()
def evaluate(
    rescore: Annotated[
        bool, typer.Option(help="Re-predict with the current rating head before scoring.")
    ] = False,
) -> None:
    """Score the rating model on approved reviews; writes reports/community_reviews.md."""
    submissions, decisions = open_store().load()
    if rescore and submissions:
        from ..assistant.embedder import Embedder
        from .rater import RatingHead

        head = RatingHead.load()
        embedder = Embedder(get_settings().data_dir / "models" / "bge-small-en-v1.5")
        predictions = head.predict(embedder.embed_passages([s.text for s in submissions]))
        for s, p in zip(submissions, np.round(predictions, 2), strict=True):
            s.model_rating, s.model_version = float(p), head.version
    text = report(submissions, latest(decisions))
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    console.print(text)
