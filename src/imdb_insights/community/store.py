"""Where submissions and moderation decisions live: one small JSON file each.

Deployed, that is a private Hugging Face dataset (a commit per submission, so nothing is lost
when Streamlit Cloud restarts the app); locally, a folder under ``data/community``. One file
per item means concurrent writers never conflict, and the history shows every change.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from .submissions import Decision, Submission

log = logging.getLogger(__name__)

REPO_TYPE = "dataset"
PATTERNS = ["submissions/**/*.json", "moderation/*.json"]

DATASET_CARD = """---
license: cc-by-4.0
viewer: false
---
# IMDb Insights AI — community reviews

Movie reviews written by visitors of the
[IMDb Insights app](https://github.com/Ahmed-Abdellmoneim/imdb-insights-ai), with the rating
they gave and the app's model prediction. Every writer agreed to this consent text
(version in each submission's `consent` field):

> {consent}

`submissions/` holds what was written; `moderation/` holds the decision on each one. The
dataset is private because unmoderated submissions can contain anything; only approved
reviews are shown in the app.
"""


class Store(Protocol):
    def add_submission(self, submission: Submission) -> None: ...

    def add_decisions(self, decisions: list[Decision]) -> None: ...

    def load(self) -> tuple[list[Submission], list[Decision]]: ...


def _write(path: Path, item: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(item.model_dump_json(indent=1), encoding="utf-8")
    tmp.replace(path)


class LocalStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def add_submission(self, submission: Submission) -> None:
        _write(self.root / submission.date_path, submission)

    def add_decisions(self, decisions: list[Decision]) -> None:
        for decision in decisions:
            _write(self.root / decision.path, decision)

    def load(self) -> tuple[list[Submission], list[Decision]]:
        def read(pattern: str, model: type[BaseModel]) -> list:
            items = []
            for path in sorted(self.root.glob(pattern)):
                try:
                    items.append(model.model_validate_json(path.read_text(encoding="utf-8")))
                except ValueError:
                    log.warning("Skipping unreadable %s", path)
            return items

        return read("submissions/*/*.json", Submission), read("moderation/*.json", Decision)


class HubStore:
    def __init__(self, repo_id: str, token: str, cache_dir: Path) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.token = token
        self.cache_dir = cache_dir
        self.api = HfApi(token=token)

    def add_submission(self, submission: Submission) -> None:
        self.api.upload_file(
            path_or_fileobj=submission.model_dump_json(indent=1).encode("utf-8"),
            path_in_repo=submission.date_path,
            repo_id=self.repo_id,
            repo_type=REPO_TYPE,
            commit_message=f"Submission for {submission.movie_title}",
        )

    def add_decisions(self, decisions: list[Decision]) -> None:
        if not decisions:
            return
        from huggingface_hub import CommitOperationAdd

        operations = [
            CommitOperationAdd(d.path, d.model_dump_json(indent=1).encode("utf-8"))
            for d in decisions
        ]
        approved = sum(d.status == "approved" for d in decisions)
        self.api.create_commit(
            self.repo_id,
            operations,
            repo_type=REPO_TYPE,
            commit_message=f"Moderation: {approved} approved, {len(decisions) - approved} rejected",
        )

    def load(self) -> tuple[list[Submission], list[Decision]]:
        from huggingface_hub import snapshot_download

        snapshot_download(
            self.repo_id,
            repo_type=REPO_TYPE,
            allow_patterns=PATTERNS,
            local_dir=self.cache_dir,
            token=self.token,
        )
        return LocalStore(self.cache_dir).load()


def create_repo(repo_id: str, token: str, consent: str) -> None:
    """Create the private dataset (idempotent) and refuse to use it if it is public."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type=REPO_TYPE).private:
        raise RuntimeError(f"{repo_id} is public; unmoderated submissions must stay private.")
    api.upload_file(
        path_or_fileobj=DATASET_CARD.format(consent=consent).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        commit_message="Dataset card",
    )


def open_store(data_dir: Path, repo_id: str | None, token: str | None) -> Store:
    """The Hugging Face store when configured, otherwise a local folder."""
    if repo_id and token:
        return HubStore(repo_id, token, data_dir / "community" / "hub")
    return LocalStore(data_dir / "community" / "local")
