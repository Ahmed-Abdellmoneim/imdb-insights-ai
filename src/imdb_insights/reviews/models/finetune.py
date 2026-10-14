"""DistilBERT fine-tuned end to end with a regression head.

Long reviews keep their first ``HEAD`` and last ``TAIL`` tokens: verdicts usually come at the
end ("...overall a 4/10"), which plain front truncation throws away (Sun et al., 2019, found
head+tail truncation best for IMDb). Training runs in fp32 because Pascal GPUs have very
low fp16 throughput. The checkpoint with the best validation MAE is kept.
"""

from __future__ import annotations

import copy
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ..metrics import point_metrics

log = logging.getLogger(__name__)

BASE_MODEL = "distilbert-base-uncased"
HEAD, TAIL = 128, 382  # + [CLS] and [SEP] = 512 tokens
CENTER, SCALE = 5.5, 4.5  # ratings 1..10 are trained as roughly -1..1


class FineTunedRegressor:
    name = "finetune"

    def __init__(
        self,
        base_model: str = BASE_MODEL,
        epochs: int = 2,
        lr: float = 3e-5,
        batch_size: int = 8,
        accumulate: int = 2,
        seed: int = 13,
        output_dir: Path | None = None,
    ) -> None:
        self.base_model = base_model
        self.epochs, self.lr, self.seed = epochs, lr, seed
        self.batch_size, self.accumulate = batch_size, accumulate
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = None

    # --- data -----------------------------------------------------------------------

    def encode(self, texts: list[str]) -> list[torch.Tensor]:
        """Token ids with head+tail truncation, wrapped as ``[CLS] ... [SEP]``."""
        ids = self.tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
        start = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id
        end = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
        out = []
        for seq in ids:
            if len(seq) > HEAD + TAIL:
                seq = seq[:HEAD] + seq[-TAIL:]
            out.append(torch.tensor([start, *seq, end]))
        return out

    def _batch(self, seqs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence(seqs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {"input_ids": input_ids.to(self.device), "attention_mask": mask.to(self.device)}

    # --- training ---------------------------------------------------------------------

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> dict:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, num_labels=1, problem_type="regression"
        ).to(self.device)
        x = self.encode(train["text"])
        y = torch.tensor(((train["rating"].to_numpy() - CENTER) / SCALE), dtype=torch.float32)

        steps_per_epoch = math.ceil(len(x) / (self.batch_size * self.accumulate))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        schedule = get_linear_schedule_with_warmup(
            optimizer, int(0.06 * steps_per_epoch * self.epochs), steps_per_epoch * self.epochs
        )
        history, best_mae, best_state, first_epoch, elapsed_before = [], math.inf, None, 1, 0.0
        checkpoint = self.output_dir / "checkpoint.pt" if self.output_dir else None
        if checkpoint and checkpoint.exists():
            # Resume after an interruption (sleep, reboot): restore everything, including the
            # RNG states, so the run continues exactly as if it had never stopped.
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            schedule.load_state_dict(state["schedule"])
            history, best_mae, best_state = state["history"], state["best_mae"], state["best_state"]
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
            if state.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda_rng"])  # dropout runs on the GPU
            first_epoch, elapsed_before = state["epoch"] + 1, state["elapsed"]
            log.info("Resuming from %s after epoch %d", checkpoint, state["epoch"])
        started = time.monotonic() - elapsed_before

        for epoch in range(first_epoch, self.epochs + 1):
            self.model.train()
            order = np.random.permutation(len(x))
            running = 0.0
            for i, start in enumerate(range(0, len(order), self.batch_size), start=1):
                idx = order[start : start + self.batch_size]
                out = self.model(**self._batch([x[j] for j in idx]))
                loss = torch.nn.functional.mse_loss(out.logits.squeeze(-1), y[idx].to(self.device))
                (loss / self.accumulate).backward()
                running += loss.item()
                if i % self.accumulate == 0 or start + self.batch_size >= len(order):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    schedule.step()
                    optimizer.zero_grad(set_to_none=True)
                if i % 200 == 0:
                    done = (epoch - 1) * len(order) + start
                    total = self.epochs * len(order)
                    eta = (time.monotonic() - started) / done * (total - done)
                    log.info("epoch %d  %d/%d  loss %.4f  eta %.0f min",
                             epoch, start, len(order), running / 200, eta / 60)  # fmt: skip
                    running = 0.0

            val = point_metrics(validation["rating"].to_numpy(), self.predict(validation["text"]))
            history.append({"epoch": epoch} | {k: round(v, 4) for k, v in val.items()})
            log.info("epoch %d validation: %s", epoch, history[-1])
            if val["mae"] < best_mae:
                best_mae = val["mae"]
                best_state = copy.deepcopy({k: v.cpu() for k, v in self.model.state_dict().items()})
            if checkpoint and epoch < self.epochs:
                self._save_checkpoint(checkpoint, epoch, optimizer, schedule, history, best_mae,
                                      best_state, time.monotonic() - started)  # fmt: skip

        self.model.load_state_dict(best_state)
        if self.output_dir:
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            if checkpoint:
                checkpoint.unlink(missing_ok=True)  # finished: the saved model supersedes it
        return {
            "base_model": Path(self.base_model).name,  # a hub id, or a local copy's folder
            "epochs": self.epochs,
            "lr": self.lr,
            "effective_batch": self.batch_size * self.accumulate,
            "truncation": f"head {HEAD} + tail {TAIL} tokens",
            "best_epoch": min(history, key=lambda h: h["mae"])["epoch"],
            "history": history,
            "train_minutes": round((time.monotonic() - started) / 60, 1),
            "device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu",
        }

    def _save_checkpoint(self, path, epoch, optimizer, schedule, history, best_mae, best_state,
                         elapsed) -> None:  # fmt: skip
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": schedule.state_dict(),
                "history": history,
                "best_mae": best_mae,
                "best_state": best_state,
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "elapsed": elapsed,
            },
            tmp,
        )
        tmp.replace(path)  # atomic: a crash mid-save never leaves a broken checkpoint
        log.info("Saved checkpoint after epoch %d", epoch)

    # --- inference --------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.model.eval()
        seqs = self.encode(list(texts))
        # Batch similar lengths together to waste less compute on padding.
        order = np.argsort([len(s) for s in seqs])
        preds = np.empty(len(seqs), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            out = self.model(**self._batch([seqs[j] for j in idx]))
            preds[idx] = out.logits.squeeze(-1).float().cpu().numpy()
        return preds * SCALE + CENTER
