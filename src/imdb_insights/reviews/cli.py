"""``uv run python -m imdb_insights.reviews {prepare,train,evaluate}``.

Models write predictions for every non-training row to ``data/reviews/predictions``; one
``evaluate`` step scores them all the same way and writes ``reports/review_models.{md,json}``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from ..config import get_settings
from ..ingest.pipeline import DB_FILENAME
from . import data, metrics

app = typer.Typer(add_completion=False, help="Review rating models (step 3).")

REPORT_DIR = Path("reports")
TRAINED_ON = "Stanford Large Movie Review Dataset, train split (movie-grouped validation held out)"
EVAL_SETS = {
    "validation": "Stanford validation (movie-grouped slice of train)",
    "test": "Stanford test, official (25,000 reviews)",
    "test_clean": "Stanford test without leaked reviews",
    "test_no_score": "Stanford test (clean) where the review does not state its x/10 score",
    "test_ood": "TMDB reviews (out of domain, includes 5-6 ratings)",
    "sample_test": "Head-to-head sample: 1,000 clean Stanford test reviews (all models)",
    "sample_ood": "Head-to-head sample: 500 TMDB reviews (all models)",
}


def _paths() -> tuple[Path, Path]:
    root = get_settings().data_dir / "reviews"
    return root / "splits.parquet", root / "predictions"


def _eval_rows(splits: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sets = {
        "validation": splits[splits["split"] == "validation"],
        "test": splits[splits["split"] == "test"],
        "test_clean": splits[(splits["split"] == "test") & ~splits["leaked"]],
        "test_no_score": splits[
            (splits["split"] == "test") & ~splits["leaked"] & ~splits["states_score"]
        ],
        "test_ood": splits[(splits["split"] == "test_ood") & ~splits["leaked"]],
    }
    sample_file = get_settings().data_dir / "reviews" / "llm_sample.parquet"
    if sample_file.exists():
        sample_ids = set(pd.read_parquet(sample_file)["review_id"])
        in_sample = splits["review_id"].isin(sample_ids)
        sets["sample_test"] = splits[in_sample & (splits["split"] == "test")]
        sets["sample_ood"] = splits[in_sample & (splits["split"] == "test_ood")]
    return sets


@app.callback()
def _setup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def prepare() -> None:
    """Build train/validation/test/test_ood splits (downloads the Stanford data once)."""
    settings = get_settings()
    stanford = data.load_stanford_cached(settings.data_dir / "external" / "stanford")
    db = settings.data_dir / "processed" / DB_FILENAME
    tmdb = data.load_tmdb(db) if db.exists() else None
    splits = data.build_splits(stanford, tmdb)
    path, _ = _paths()
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(path, index=False)
    summary = splits.groupby(["split", "leaked"]).size().unstack(fill_value=0)
    typer.echo(f"Wrote {path}\n{summary}")


@app.command()
def train(
    model_name: Annotated[str, typer.Argument(help="tfidf, embeddings or finetune")],
    base_model: Annotated[
        str | None,
        typer.Option(help="finetune only: Hugging Face id or local folder of the base model."),
    ] = None,
) -> None:
    """Fit a model on train (tuned on validation) and save its predictions."""
    from .models import get_model

    splits_path, pred_dir = _paths()
    splits = pd.read_parquet(splits_path)
    options = {"base_model": base_model} if model_name == "finetune" and base_model else {}
    model = get_model(model_name, **options)
    if model_name == "finetune":
        model.output_dir = get_settings().data_dir / "reviews" / "models" / "finetune"
    info = model.fit(splits[splits["split"] == "train"], splits[splits["split"] == "validation"])
    if hasattr(model, "top_terms"):
        info["top_terms"] = model.top_terms()
    if hasattr(model, "head"):
        from ..community.rater import save_head

        mae = info["validation_mae_by_alpha"][info["alpha"]]
        path = save_head(model.head() | {"validation_mae": mae, "trained_on": TRAINED_ON})
        typer.echo(f"Saved the rating head used by the app to {path}")

    rows = splits[splits["split"] != "train"]
    preds = pd.DataFrame(
        {"review_id": rows["review_id"].to_numpy(), "prediction": model.predict(rows["text"])}
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(pred_dir / f"{model.name}.parquet", index=False)
    (pred_dir / f"{model.name}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    typer.echo(json.dumps(info, indent=2))


@app.command()
def llm(
    model_dir: Annotated[
        Path, typer.Option(help="Local folder of an instruction-tuned causal LM.")
    ] = Path("data/reviews/models/Qwen2.5-0.5B-Instruct"),
) -> None:
    """Zero-shot ratings from a local open-weights LLM on the head-to-head sample (free).

    Writes two prediction sets: the raw expected score, and the same scores mapped onto the
    dataset's scale with a line fitted on the sample's validation reviews.
    """
    from . import llm as local_llm

    splits_path, pred_dir = _paths()
    sample = local_llm.build_sample(pd.read_parquet(splits_path))
    path = local_llm.sample_path(get_settings().data_dir)
    sample[["review_id"]].to_parquet(path, index=False)

    rater = local_llm.LocalLLMRater(model_dir)
    started = time.monotonic()
    expected, mass = rater.predict(sample["text"].tolist())
    minutes = (time.monotonic() - started) / 60

    is_val = (sample["split"] == "validation").to_numpy()
    slope, intercept = local_llm.calibrate(expected[is_val], sample["rating"].to_numpy()[is_val])
    eval_ids = sample.loc[~is_val, "review_id"].to_numpy()
    raw, calibrated = expected[~is_val], slope * expected[~is_val] + intercept
    info = {
        "model": rater.name,
        "method": "zero-shot; expected score from next-token probabilities of 1..10",
        "reviews": int((~is_val).sum()),
        "mean_answer_mass": round(float(mass.mean()), 4),
        "seconds_per_review": round(minutes * 60 / len(sample), 3),
        "device": rater.device,
    }
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, values, extra in [
        ("llm-zero-shot", raw, {}),
        ("llm-calibrated", calibrated, {"calibration": {"slope": round(slope, 4),
                                        "intercept": round(intercept, 4),
                                        "fitted_on_validation_reviews": int(is_val.sum())}}),
    ]:  # fmt: skip
        pd.DataFrame({"review_id": eval_ids, "prediction": values}).to_parquet(
            pred_dir / f"{name}.parquet", index=False
        )
        (pred_dir / f"{name}.json").write_text(json.dumps(info | extra, indent=2), "utf-8")
    typer.echo(json.dumps(info | {"calibration": [slope, intercept]}, indent=2))


@app.command()
def evaluate() -> None:
    """Score every saved model (plus a predict-the-mean baseline) on all evaluation sets."""
    splits_path, pred_dir = _paths()
    splits = pd.read_parquet(splits_path)
    sets = _eval_rows(splits)
    train_mean = float(splits.loc[splits["split"] == "train", "rating"].mean())

    predictions = {"mean baseline": None}
    predictions |= {p.stem: pd.read_parquet(p) for p in sorted(pred_dir.glob("*.parquet"))}
    results: dict[str, dict] = {}
    for name, preds in predictions.items():
        results[name] = {"sets": {}, "info": {}}
        info_file = pred_dir / f"{name}.json"
        if info_file.exists():
            results[name]["info"] = json.loads(info_file.read_text(encoding="utf-8"))
        for set_name, rows in sets.items():
            if rows.empty:
                continue
            if preds is None:
                y_pred = np.full(len(rows), train_mean)
            else:
                merged = rows.merge(preds, on="review_id", how="left")
                coverage = merged["prediction"].notna().mean()
                if coverage < 0.95:
                    continue  # model was not run on this set (e.g. the LLM outside the sample)
                # A few missing answers (refusals, errors) count as a neutral guess rather
                # than being dropped, so a model cannot look better by skipping hard cases.
                y_pred = merged["prediction"].fillna(train_mean).to_numpy()
            scores = metrics.evaluate(rows["rating"].to_numpy(), y_pred)
            if preds is not None:
                scores["coverage"] = round(float(coverage), 4)
            scores["per_rating_mae"] = metrics.per_rating_mae(rows["rating"].to_numpy(), y_pred)
            results[name]["sets"][set_name] = scores

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "review_models.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (REPORT_DIR / "review_models.md").write_text(render_markdown(results), encoding="utf-8")
    typer.echo(render_markdown(results))


def render_markdown(results: dict[str, dict]) -> str:
    lines = ["# Review rating models", ""]
    for set_name, title in EVAL_SETS.items():
        rows = [(n, r["sets"][set_name]) for n, r in results.items() if set_name in r["sets"]]
        if not rows:
            continue
        n = rows[0][1]["n"]
        lines += [
            f"## {title}", "", f"{n:,} reviews. 95% bootstrap intervals in brackets.", "",
            "| model | MAE ↓ | RMSE ↓ | Spearman ↑ | within ±1 ↑ | polarity acc. ↑ | bias |",
            "|---|---|---|---|---|---|---|",
        ]  # fmt: skip
        for name, s in rows:
            lines.append(
                f"| {name} | {s['mae']:.3f} [{s['mae_lo']:.3f}–{s['mae_hi']:.3f}] "
                f"| {s['rmse']:.3f} | {s['spearman']:.3f} | {s['within_1']:.1%} "
                f"| {s['polarity_accuracy']:.1%} [{s['polarity_accuracy_lo']:.1%}–"
                f"{s['polarity_accuracy_hi']:.1%}] | {s['bias']:+.2f} |"
            )
        lines.append("")
    return "\n".join(lines).replace("nan%", "–").replace("| nan |", "| – |")
