"""Zero-shot rating prediction with Claude, via the Message Batches API.

No training: the model reads each review and returns the rating it believes the author
gave, as schema-validated JSON. Run on a stratified sample (the same sample every model is
scored on), submitted as one batch (50% of standard prices; most batches finish within an
hour). Refusals and errors are counted, never silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

MODEL = "claude-opus-5-5"
EFFORT = "low"  # judging a review's sentiment needs little deliberation
MAX_TOKENS = 2048  # thinking counts toward max_tokens; the answer itself is ~10 tokens
# Batch prices per million tokens for MODEL (50% of $4 input / $20 output).
BATCH_PRICE_IN, BATCH_PRICE_OUT = 2.0, 10.0

SYSTEM = (
    "You are rating movie reviews. Each review was written by a user of a movie site who also "
    "gave the movie a score from 1 (worst) to 10 (best). Read the review and estimate the "
    "score its author gave. Judge the author's overall opinion of the movie, not your own. "
    "Answer with the single most likely integer score."
)
SCHEMA = {
    "type": "object",
    "properties": {"rating": {"type": "integer", "minimum": 1, "maximum": 10}},
    "required": ["rating"],
    "additionalProperties": False,
}


def stratified_sample(rows: pd.DataFrame, n: int, seed: int = 13) -> pd.DataFrame:
    """``n`` rows with the same rating mix as ``rows`` (proportional allocation)."""
    if len(rows) <= n:
        return rows
    shares = rows["rating"].value_counts(normalize=True)
    quotas = (shares * n).round().astype(int)
    parts = [
        rows[rows["rating"] == rating].sample(
            min(q, (rows["rating"] == rating).sum()), random_state=seed
        )  # fmt: skip
        for rating, q in quotas.items()
    ]
    return pd.concat(parts)


def build_sample(splits: pd.DataFrame, n_test: int = 1000, n_ood: int = 500) -> pd.DataFrame:
    clean_test = splits[(splits["split"] == "test") & ~splits["leaked"]]
    clean_ood = splits[(splits["split"] == "test_ood") & ~splits["leaked"]]
    return pd.concat([stratified_sample(clean_test, n_test), stratified_sample(clean_ood, n_ood)])


def build_requests(sample: pd.DataFrame) -> list[Request]:
    return [
        Request(
            custom_id=review_id,
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM,
                output_config={
                    "effort": EFFORT,
                    "format": {"type": "json_schema", "schema": SCHEMA},
                },  # fmt: skip
                messages=[{"role": "user", "content": f"<review>\n{text}\n</review>"}],
            ),
        )
        for review_id, text in zip(sample["review_id"], sample["text"], strict=True)
    ]


def submit(client: anthropic.Anthropic, sample: pd.DataFrame) -> str:
    # Batch custom_ids allow 1-64 characters from [a-zA-Z0-9_-].
    return client.messages.batches.create(requests=build_requests(sample)).id


@dataclass
class Collected:
    predictions: pd.DataFrame
    info: dict


def collect(client: anthropic.Anthropic, batch_id: str) -> Collected | None:
    """Results once the batch has ended (``None`` while it is still running)."""
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        return None
    rows, outcomes = [], {"succeeded": 0, "refused": 0, "unparseable": 0, "errored": 0}
    tokens_in = tokens_out = 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            outcomes["errored"] += 1
            continue
        msg = result.result.message
        tokens_in += msg.usage.input_tokens
        tokens_out += msg.usage.output_tokens
        if msg.stop_reason == "refusal":
            outcomes["refused"] += 1
            continue
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            rating = int(json.loads(text)["rating"])
        except (ValueError, KeyError, TypeError):
            outcomes["unparseable"] += 1
            continue
        outcomes["succeeded"] += 1
        rows.append({"review_id": result.custom_id, "prediction": float(rating)})
    cost = tokens_in / 1e6 * BATCH_PRICE_IN + tokens_out / 1e6 * BATCH_PRICE_OUT
    info = {
        "model": MODEL,
        "effort": EFFORT,
        "api": "Message Batches (50% price)",
        "batch_id": batch_id,
        "outcomes": outcomes,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cost_usd": round(cost, 2),
        "cost_per_1k_reviews_usd": round(cost / max(1, sum(outcomes.values())) * 1000, 2),
    }
    return Collected(pd.DataFrame(rows, columns=["review_id", "prediction"]), info)


def estimate_cost(sample: pd.DataFrame, thinking_tokens: int = 300) -> float:
    """Rough upper estimate before submitting (~1.4 tokens per word)."""
    tokens_in = (len(SYSTEM.split()) + sample["text"].str.split().str.len() + 10) * 1.4
    tokens_out = len(sample) * (thinking_tokens + 15)
    return float(tokens_in.sum() / 1e6 * BATCH_PRICE_IN + tokens_out / 1e6 * BATCH_PRICE_OUT)


def sample_path(data_dir: Path) -> Path:
    return data_dir / "reviews" / "llm_sample.parquet"
