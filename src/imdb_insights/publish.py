"""Share the warehouse through a private Hugging Face dataset.

The dataset holds the DuckDB warehouse and manifest (read by the deployed app) and a
compressed copy of the API response cache (restored by the scheduled refresh, so each run
only fetches what changed). TMDB's terms do not allow redistributing its data in bulk, so
publishing refuses to write to a dataset that is public. Each push squashes the dataset's
history, keeping only the latest version.

Usage: ``uv run python -m imdb_insights.publish {push,pull-cache,pull}``
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

import typer
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from .config import get_settings
from .ingest.pipeline import DB_FILENAME

log = logging.getLogger(__name__)

REPO_TYPE = "dataset"
MANIFEST = "manifest.json"
CACHE_ARCHIVE = "raw-cache.tar.gz"
WAREHOUSE_FILES = (DB_FILENAME, MANIFEST)
# Discover results are dated snapshots that are only reused on the same day.
CACHE_EXCLUDE = ("tmdb/discover",)

DATASET_CARD = """---
license: other
viewer: false
---
# IMDb Insights AI — private warehouse

Built by [imdb-insights-ai](https://github.com/Ahmed-Abdellmoneim/imdb-insights-ai).
Kept private: it contains data from the TMDB API, which may not be redistributed in bulk.
This product uses the TMDB API but is not endorsed or certified by TMDB.
"""


class PublishError(RuntimeError):
    pass


def pack_cache(raw_dir: Path, archive: Path) -> None:
    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        relative = Path(info.name).relative_to("raw").as_posix() if info.name != "raw" else ""
        return None if relative.startswith(CACHE_EXCLUDE) else info

    with tarfile.open(archive, "w:gz") as tar:
        tar.add(raw_dir, arcname="raw", filter=keep)


def unpack_cache(archive: Path, data_dir: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        # The "data" filter rejects absolute paths, links outside the target, etc.
        tar.extractall(data_dir, filter="data")


def _api(token: str) -> HfApi:
    return HfApi(token=token)


def push(data_dir: Path, repo_id: str, token: str) -> None:
    processed = data_dir / "processed"
    missing = [f for f in WAREHOUSE_FILES if not (processed / f).exists()]
    if missing:
        raise PublishError(f"Nothing to publish: {', '.join(missing)} missing in {processed}")

    api = _api(token)
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type=REPO_TYPE).private:
        raise PublishError(
            f"{repo_id} is public. TMDB data must not be redistributed; make it private first."
        )

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in WAREHOUSE_FILES:
            shutil.copy2(processed / name, stage / name)
        if (data_dir / "raw").exists():
            pack_cache(data_dir / "raw", stage / CACHE_ARCHIVE)
        (stage / "README.md").write_text(DATASET_CARD, encoding="utf-8")
        api.upload_folder(
            folder_path=stage,
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            commit_message="Data refresh",
        )
    api.super_squash_history(repo_id, repo_type=REPO_TYPE)
    log.info("Published %s to %s", ", ".join(WAREHOUSE_FILES), repo_id)


def _download(repo_id: str, token: str, filename: str, target_dir: Path) -> Path | None:
    """Download one file into ``target_dir``; ``None`` if the dataset or file is absent."""
    try:
        return Path(
            hf_hub_download(
                repo_id, filename, repo_type=REPO_TYPE, token=token, local_dir=target_dir
            )
        )
    except (RepositoryNotFoundError, EntryNotFoundError):
        return None


def pull_cache(data_dir: Path, repo_id: str, token: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        archive = _download(repo_id, token, CACHE_ARCHIVE, Path(tmp))
        if archive is None:
            log.info("No published cache yet; starting from scratch")
            return False
        unpack_cache(archive, data_dir)
    log.info("Restored the API cache from %s", repo_id)
    return True


def pull_warehouse(data_dir: Path, repo_id: str, token: str) -> bool:
    """Download the published warehouse into ``<data_dir>/processed``.

    Files land in a temporary folder first and are then moved into place, so a reader
    never sees a half-downloaded database.
    """
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=processed) as tmp:
        files = [_download(repo_id, token, name, Path(tmp)) for name in WAREHOUSE_FILES]
        if any(f is None for f in files):
            return False
        for file in files:
            file.replace(processed / file.name)
    return True


# --- CLI ------------------------------------------------------------------------------

app = typer.Typer(add_completion=False, help="Publish or restore data via Hugging Face.")


def _target() -> tuple[Path, str, str]:
    settings = get_settings()
    if not settings.hf_dataset_repo or not settings.hf_token:
        typer.echo("Set HF_DATASET_REPO and HF_TOKEN (see .env.example).", err=True)
        raise typer.Exit(code=2)
    return settings.data_dir, settings.hf_dataset_repo, settings.hf_token.get_secret_value()


@app.callback()
def _setup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HTTP request is noise


@app.command("push")
def push_command() -> None:
    """Upload the warehouse and the API cache."""
    push(*_target())


@app.command("pull-cache")
def pull_cache_command() -> None:
    """Restore the API cache (no-op on the very first run)."""
    pull_cache(*_target())


@app.command("pull")
def pull_command() -> None:
    """Download the published warehouse."""
    if not pull_warehouse(*_target()):
        typer.echo("No published warehouse yet.", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app(prog_name="imdb-publish")
