"""bge-small-en-v1.5 text embeddings with ONNX Runtime (CPU, no PyTorch).

The official ONNX export is pinned by SHA-256 and downloaded once. Embeddings follow the
model card: the [CLS] token's final hidden state, L2-normalized, so a dot product is the
cosine similarity. Retrieval queries get the model's recommended instruction prefix;
passages do not. ``tests`` and the step 3 reference implementation (sentence-transformers)
agree to within float rounding.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np

REPO = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # pinned commit (also for the hash)
FILES = {"model.onnx": "onnx/model.onnx", "tokenizer.json": "tokenizer.json"}
MODEL_SHA256 = "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
DIMENSIONS = 384
MAX_TOKENS = 512
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def ensure_model(model_dir: Path) -> Path:
    """Download the pinned model files if missing, and verify the model's checksum."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for local_name, repo_path in FILES.items():
        if not (model_dir / local_name).exists():
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(REPO, repo_path, revision=REVISION)
            shutil.copyfile(downloaded, model_dir / local_name)
    digest = hashlib.sha256((model_dir / "model.onnx").read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError(f"{model_dir / 'model.onnx'} has unexpected SHA-256 {digest}")
    return model_dir


class Embedder:
    def __init__(self, model_dir: Path, *, threads: int | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        ensure_model(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def _encode(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in batch], dtype=np.int64)
        mask = np.array([e.attention_mask for e in batch], dtype=np.int64)
        hidden = self.session.run(
            ["last_hidden_state"],
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)},
        )[0]
        cls = hidden[:, 0, :]
        return cls / np.linalg.norm(cls, axis=1, keepdims=True)

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, DIMENSIONS), dtype=np.float32)
        # Sorting by length keeps padding (wasted compute) low; results are put back in order.
        order = np.argsort([len(t) for t in texts])
        out = np.empty((len(texts), DIMENSIONS), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            idx = order[start : start + batch_size]
            out[idx] = self._encode([texts[i] for i in idx])
        return out

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([QUERY_PREFIX + query])[0].astype(np.float32)
