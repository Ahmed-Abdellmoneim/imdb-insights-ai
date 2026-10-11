import tarfile
from types import SimpleNamespace

import pytest

from imdb_insights import publish


def _make_data(tmp_path):
    raw = tmp_path / "raw"
    for rel in ("omdb/tt1.json", "tmdb/movie/1.json", "tmdb/discover/2026-01-01/q/1.json"):
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        (raw / rel).write_text("{}", encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    for name in publish.WAREHOUSE_FILES:
        (processed / name).write_text("x", encoding="utf-8")
    return tmp_path


def test_cache_round_trip_skips_dated_discover_pages(tmp_path):
    data = _make_data(tmp_path / "src")
    archive = tmp_path / "cache.tar.gz"
    publish.pack_cache(data / "raw", archive)

    restored = tmp_path / "restored"
    publish.unpack_cache(archive, restored)
    files = sorted(p.relative_to(restored).as_posix() for p in restored.rglob("*.json"))
    assert files == ["raw/omdb/tt1.json", "raw/tmdb/movie/1.json"]


def test_unpack_rejects_paths_outside_the_target(tmp_path):
    evil = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("x", encoding="utf-8")
    with tarfile.open(evil, "w:gz") as tar:
        tar.add(payload, arcname="../escaped.txt")
    with pytest.raises(tarfile.TarError):
        publish.unpack_cache(evil, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


class FakeApi:
    def __init__(self, private=True):
        self.private = private
        self.uploaded = None
        self.squashed = False

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        assert private is True and exist_ok is True

    def repo_info(self, repo_id, repo_type):
        return SimpleNamespace(private=self.private)

    def upload_folder(self, folder_path, repo_id, repo_type, commit_message):
        self.uploaded = sorted(p.name for p in folder_path.iterdir())

    def super_squash_history(self, repo_id, repo_type):
        self.squashed = True


def test_push_uploads_warehouse_cache_and_card(tmp_path, monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(publish, "_api", lambda token: api)
    publish.push(_make_data(tmp_path), "me/data", "token")
    assert api.uploaded == sorted([*publish.WAREHOUSE_FILES, publish.CACHE_ARCHIVE, "README.md"])
    assert api.squashed


def test_push_refuses_a_public_dataset(tmp_path, monkeypatch):
    api = FakeApi(private=False)
    monkeypatch.setattr(publish, "_api", lambda token: api)
    with pytest.raises(publish.PublishError, match="public"):
        publish.push(_make_data(tmp_path), "me/data", "token")
    assert api.uploaded is None


def test_push_requires_a_built_warehouse(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "_api", lambda token: FakeApi())
    with pytest.raises(publish.PublishError, match="missing"):
        publish.push(tmp_path, "me/data", "token")


def test_pull_warehouse_moves_files_into_place(tmp_path, monkeypatch):
    def fake_download(repo_id, token, filename, target_dir):
        path = target_dir / filename
        path.write_text(f"new {filename}", encoding="utf-8")
        return path

    monkeypatch.setattr(publish, "_download", fake_download)
    assert publish.pull_warehouse(tmp_path, "me/data", "token")
    processed = tmp_path / "processed"
    assert sorted(p.name for p in processed.iterdir()) == sorted(publish.WAREHOUSE_FILES)


def test_pull_warehouse_reports_nothing_published(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "_download", lambda *a: None)
    assert publish.pull_warehouse(tmp_path, "me/data", "token") is False
