"""models.lock.json: what may be pinned, how a local copy is checked, and the CLI around it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from raglab import models
from raglab.cli import app
from raglab.models import ModelError, ModelLock, ModelSpec, problems, require
from conftest import REAL_MODELS_LOCK

REVISION = "0123456789abcdef0123456789abcdef01234567"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_lock(path: Path, models_: dict, schema: int = 1) -> Path:
    path.write_text(json.dumps({"schema": schema, "models": models_}), encoding="utf-8")
    return path


def entry(files: dict | None = None, **overrides) -> dict:
    e = {"repo": "owner/model", "revision": REVISION, "license": "MIT",
         "files": files if files is not None else {"model.safetensors": sha(b"weights"), "config.json": sha(b"{}")}}
    e.update(overrides)
    return e


# ---------------------------------------------------------------- the committed lock


def test_the_committed_lock_pins_both_roles_by_full_commit_sha():
    lock = ModelLock.load(REAL_MODELS_LOCK)
    assert set(lock.models) == {"embedder", "reranker"}
    assert lock["embedder"].repo == "BAAI/bge-small-en-v1.5"
    assert lock["reranker"].repo == "cross-encoder/ms-marco-MiniLM-L6-v2"
    for spec in lock.models.values():
        assert len(spec.revision) == 40
        assert "model.safetensors" in spec.files
        assert all(name.endswith((".safetensors", ".json", ".txt")) for name in spec.files)


def test_a_lock_survives_a_save_and_load(tmp_path):
    lock = ModelLock.load(REAL_MODELS_LOCK)
    lock.save(tmp_path / "models.lock.json")
    again = ModelLock.load(tmp_path / "models.lock.json")
    assert {r: s.fingerprint for r, s in again.models.items()} == {r: s.fingerprint for r, s in lock.models.items()}
    assert (tmp_path / "models.lock.json").read_text() == REAL_MODELS_LOCK.read_text()


def test_the_fingerprint_moves_with_the_revision_and_every_file():
    spec = ModelSpec("embedder", "owner/model", REVISION, "MIT", {"model.safetensors": sha(b"a")})
    assert spec.fingerprint != ModelSpec("embedder", "owner/model", "f" * 40, "MIT", spec.files).fingerprint
    assert spec.fingerprint != ModelSpec("embedder", "owner/model", REVISION, "MIT",
                                         {"model.safetensors": sha(b"b")}).fingerprint
    assert spec.fingerprint != ModelSpec("embedder", "other/model", REVISION, "MIT", spec.files).fingerprint


# ---------------------------------------------------------------- what a lock may say


@pytest.mark.parametrize("bad, message", [
    (entry(revision="main"), "full 40-character commit SHA"),
    (entry(revision=REVISION[:12]), "full 40-character commit SHA"),
    (entry(files={"pytorch_model.bin": sha(b"x"), "model.safetensors": sha(b"y")}), "not an allowed file type"),
    (entry(files={"model.safetensors": sha(b"y"), "modeling_custom.py": sha(b"z")}), "not an allowed file type"),
    (entry(files={"../escape.json": sha(b"x"), "model.safetensors": sha(b"y")}), "plain relative path"),
    (entry(files={"/abs/config.json": sha(b"x"), "model.safetensors": sha(b"y")}), "plain relative path"),
    (entry(files={"config.json": sha(b"x")}), "no safetensors weights"),
    (entry(files={"model.safetensors": "abc"}), "64-character lowercase SHA-256"),
    (entry(files={"model.safetensors": sha(b"x").upper()}), "64-character lowercase SHA-256"),
    (entry(repo="not a repo"), "owner/name"),
    (entry(license=""), "license must be named"),
    ({"repo": "owner/model", "revision": REVISION, "files": {}}, "exactly repo, revision, license and files"),
])
def test_a_lock_that_could_load_the_wrong_thing_is_refused(tmp_path, bad, message):
    with pytest.raises(ModelError, match=message):
        ModelLock.load(write_lock(tmp_path / "models.lock.json", {"embedder": bad}))


def test_unknown_roles_schemas_and_keys_are_refused(tmp_path):
    with pytest.raises(ModelError, match="not a known role"):
        ModelLock.load(write_lock(tmp_path / "a.json", {"generator": entry()}))
    with pytest.raises(ModelError, match="schema-1"):
        ModelLock.load(write_lock(tmp_path / "b.json", {"embedder": entry()}, schema=2))
    (tmp_path / "c.json").write_text(json.dumps({"schema": 1, "models": {}, "extra": 1}))
    with pytest.raises(ModelError, match="unexpected keys"):
        ModelLock.load(tmp_path / "c.json")
    (tmp_path / "d.json").write_text("{ not json")
    with pytest.raises(ModelError, match="not valid JSON"):
        ModelLock.load(tmp_path / "d.json")
    with pytest.raises(ModelError, match="Restore models.lock.json"):
        ModelLock.load(tmp_path / "missing.json")
    with pytest.raises(ModelError, match="pins no 'reranker'"):
        ModelLock.load(write_lock(tmp_path / "e.json", {"embedder": entry()}))["reranker"]


# ---------------------------------------------------------------- checking a local copy


def local_copy(tmp_path: Path) -> tuple[ModelSpec, Path]:
    folder = tmp_path / "snapshot"
    (folder / "1_Pooling").mkdir(parents=True)
    files = {"model.safetensors": b"weights", "config.json": b"{}", "1_Pooling/config.json": b'{"pool": 1}'}
    for name, data in files.items():
        (folder / name).write_bytes(data)
    return ModelSpec("embedder", "owner/model", REVISION, "MIT", {n: sha(d) for n, d in files.items()}), folder


def test_a_matching_copy_has_no_problems_and_every_difference_is_named(tmp_path):
    spec, folder = local_copy(tmp_path)
    assert problems(spec, folder) == []
    (folder / "config.json").write_bytes(b'{"changed": true}')
    (folder / "1_Pooling" / "config.json").unlink()
    assert problems(spec, folder) == ["1_Pooling/config.json: missing",
                                      "config.json: SHA-256 differs from models.lock.json"]


def test_require_refuses_a_missing_or_altered_copy(tmp_path, monkeypatch):
    spec, folder = local_copy(tmp_path)
    monkeypatch.setattr(models, "local_dir", lambda s: None)
    with pytest.raises(ModelError, match="models fetch"):
        require(spec)
    monkeypatch.setattr(models, "local_dir", lambda s: folder)
    assert require(spec) == folder
    (folder / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ModelError, match="doesn't match models.lock.json: model.safetensors"):
        require(spec)


def test_pin_refuses_a_branch_name():
    with pytest.raises(ModelError, match="full 40-character commit SHA"):
        models.pin("embedder", "owner/model", "main", "MIT")


# ---------------------------------------------------------------- the CLI


def test_models_verify_reports_each_role(tmp_path, monkeypatch):
    spec, folder = local_copy(tmp_path)
    lock_path = tmp_path / "models.lock.json"
    ModelLock({"embedder": spec, "reranker": ModelSpec("reranker", "owner/rr", REVISION, "MIT", spec.files)}).save(lock_path)
    monkeypatch.setattr("raglab.cli.default_models_lock_path", lambda: lock_path)
    monkeypatch.setattr("raglab.cli.local_dir", lambda s: folder if s.role == "embedder" else None)
    result = CliRunner().invoke(app, ["models", "verify"])
    assert result.exit_code == 1
    assert "OK embedder" in result.output and "not downloaded" in result.output and "models fetch" in result.output
    monkeypatch.setattr("raglab.cli.local_dir", lambda s: folder)
    result = CliRunner().invoke(app, ["models", "verify"])
    assert result.exit_code == 0, result.output
    assert result.output.count("files match the lock") == 2


# ---------------------------------------------------------------- the real models


@pytest.mark.models
def test_the_downloaded_models_match_the_lock(real_models):
    lock = ModelLock.load(REAL_MODELS_LOCK)
    for role, folder in real_models.items():
        assert problems(lock[role], folder) == []


@pytest.mark.models
def test_the_real_embedder_is_deterministic_per_text(real_models):
    from raglab.neural import QUERY_INSTRUCTION, SentenceTransformerEmbedder

    spec = ModelLock.load(REAL_MODELS_LOCK)["embedder"]
    embedder = SentenceTransformerEmbedder(spec, real_models["embedder"])
    texts = ["Use HTTPException to return a 404.", "Declare a request body with a Pydantic model.", "x"]
    together = embedder.embed(texts, "")
    alone = np.stack([embedder.embed([t], "")[0] for t in texts])
    assert together.shape == (3, 384) and together.dtype == np.float32
    assert np.array_equal(together, alone)  # a vector never depends on its batch
    assert np.allclose(np.linalg.norm(together, axis=1), 1.0, atol=1e-5)
    query = embedder.embed(["return a 404 error"], QUERY_INSTRUCTION)[0]
    assert not np.array_equal(query, embedder.embed(["return a 404 error"], "")[0])  # the instruction is used
    assert int(np.argmax(together[:2] @ query)) == 0
    assert embedder.settings["threads"] == 4 and embedder.settings["batch_size"] == 1


@pytest.mark.models
def test_the_real_reranker_prefers_the_relevant_passage(real_models):
    from raglab.neural import CrossEncoderScorer

    spec = ModelLock.load(REAL_MODELS_LOCK)["reranker"]
    scorer = CrossEncoderScorer(spec, real_models["reranker"])
    passages = ["Raise HTTPException with status_code=404 to return a not found error.",
                "Static files are served with StaticFiles mounted on the app."]
    scores = scorer.score("How do I return a 404 error?", passages)
    assert scores.shape == (2,) and scores[0] > scores[1]
    assert np.array_equal(scores, scorer.score("How do I return a 404 error?", passages))
