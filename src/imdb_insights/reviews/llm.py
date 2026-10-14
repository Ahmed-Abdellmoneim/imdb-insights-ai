"""Zero-shot rating prediction with a small open-weights LLM running locally (free).

Instead of letting the model write an answer and parsing it, we read its next-token
probabilities for the scores 1-10 and take the probability-weighted mean. There is no
parsing failure, and the prediction is a calibrated-looking expected value suited to MAE.

Digits are single tokens in Qwen's vocabulary, so "10" is the token "1" followed by "0":

    P(10) = P("1") * P("0" | "1")        P(1) = P("1") * (1 - P("0" | "1"))
    P(k)  = P("k") for k = 2..9

and the ten probabilities are renormalized (the mass the model puts on any other token is
reported as ``answer_mass``, a check that it understood the instruction).

Everything heavy (torch, transformers) is imported lazily so the sampling helpers can be
used and tested without the optional ``ml`` dependency group.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

SYSTEM = (
    "You read a movie review and estimate the score from 1 (worst) to 10 (best) that its "
    "author gave the movie. Judge the author's overall opinion, not your own. "
    "Reply with only the number."
)
MAX_REVIEW_TOKENS = 768  # head + tail of longer reviews, like the fine-tuned model
SCORES = np.arange(1, 11)


# --- sampling -----------------------------------------------------------------------------


def stratified_sample(rows: pd.DataFrame, n: int, seed: int = 13) -> pd.DataFrame:
    """``n`` rows with the same rating mix as ``rows`` (proportional allocation)."""
    if len(rows) <= n:
        return rows
    quotas = (rows["rating"].value_counts(normalize=True) * n).round().astype(int)
    parts = [
        rows[rows["rating"] == rating].sample(
            min(q, int((rows["rating"] == rating).sum())), random_state=seed
        )  # fmt: skip
        for rating, q in quotas.items()
    ]
    return pd.concat(parts)


def build_sample(
    splits: pd.DataFrame, n_test: int = 1000, n_ood: int = 500, n_validation: int = 300
) -> pd.DataFrame:
    """Head-to-head evaluation sample, plus a validation slice for calibration."""
    clean_test = splits[(splits["split"] == "test") & ~splits["leaked"]]
    clean_ood = splits[(splits["split"] == "test_ood") & ~splits["leaked"]]
    validation = splits[splits["split"] == "validation"]
    return pd.concat(
        [
            stratified_sample(clean_test, n_test),
            stratified_sample(clean_ood, n_ood),
            stratified_sample(validation, n_validation),
        ]
    )


def sample_path(data_dir: Path) -> Path:
    return data_dir / "reviews" / "llm_sample.parquet"


def calibrate(raw: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Least-squares line mapping the LLM's expected score onto the dataset's scale."""
    slope, intercept = np.polyfit(raw, truth, 1)
    return float(slope), float(intercept)


# --- the rater ------------------------------------------------------------------------


class LocalLLMRater:
    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # fp32: Pascal GPUs have very low fp16 throughput; 0.5B parameters fit in 4 GB.
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
        self.model.to(self.device).eval()
        self.name = Path(model_dir).name
        self.digit_ids = []
        for d in range(10):
            ids = self.tokenizer.encode(str(d), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Digit {d} is not a single token for {self.name}")
            self.digit_ids.append(ids[0])

    def _prompt(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > MAX_REVIEW_TOKENS:
            head = MAX_REVIEW_TOKENS // 4
            ids = ids[:head] + ids[-(MAX_REVIEW_TOKENS - head) :]
        review = self.tokenizer.decode(ids)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Review:\n{review}"},
        ]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )

    def score_distribution(self, text: str) -> tuple[np.ndarray, float]:
        """Probabilities of scores 1..10 (renormalized) and the raw mass on valid answers."""
        torch = self.torch
        with torch.no_grad():
            ids = torch.tensor([self._prompt(text)], device=self.device)
            first = self.model(input_ids=ids, use_cache=True)
            p = torch.softmax(first.logits[0, -1].float(), dim=-1)
            # Second step only for "1": is it "1" or the start of "10"?
            one = torch.tensor([[self.digit_ids[1]]], device=self.device)
            second = self.model(input_ids=one, past_key_values=first.past_key_values)
            p_zero_after_one = torch.softmax(second.logits[0, -1].float(), dim=-1)[
                self.digit_ids[0]
            ].item()
        digits = p[self.digit_ids].cpu().numpy()
        probs = np.empty(10)
        probs[0] = digits[1] * (1 - p_zero_after_one)  # "1"
        probs[1:9] = digits[2:10]  # "2".."9"
        probs[9] = digits[1] * p_zero_after_one  # "10"
        mass = float(probs.sum())
        return probs / mass, mass

    def predict(self, texts: list[str], log_every: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Expected scores and the per-review answer mass."""
        expected, masses = np.empty(len(texts)), np.empty(len(texts))
        started = time.monotonic()
        for i, text in enumerate(texts):
            probs, masses[i] = self.score_distribution(text)
            expected[i] = float(probs @ SCORES)
            if log_every and (i + 1) % log_every == 0:
                rate = (time.monotonic() - started) / (i + 1)
                print(f"{i + 1}/{len(texts)}  {rate:.2f}s/review  "
                      f"eta {rate * (len(texts) - i - 1) / 60:.0f} min", flush=True)  # fmt: skip
        return expected, masses
