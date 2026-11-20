"""models.lock.json: what may be pinned, how a local copy is fetched and checked, and the CLI around it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from typer.testing import CliRunner

from conftest import REAL_MODELS_LOCK
from raglab import models
from raglab.cli import app
from raglab.models import (
    LockedFile, ModelError, ModelLock, ModelSpec, ensure_not_synced, fetch, problems, require,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def locked(data: bytes) -> dict:
    return {"sha256": sha(data), "bytes": len(data)}


def write_lock(path: Path, models_: dict, schema: int = 1) -> Path:
    path.write_text(json.dumps({"schema": schema, "models": models_}), encoding="utf-8")
    return path


def entry(files: dict | None = None, **overrides) -> dict:
    e = {"repo": "owner/model", "revision": REVISION, "license": "MIT",
         "files": files if files is not None else {"model.safetensors": locked(b"weights"), "config.json": locked(b"{}")}}
    e.update(overrides)
    return e


FILES = {"model.safetensors": b"weights", "config.json": b"{}", "1_Pooling/config.json": b'{"pool": 1}'}


def spec_for(files: dict[str, bytes] = FILES, role: str = "embedder") -> ModelSpec:
    return ModelSpec(role, "owner/model", REVISION, "MIT",
                     MappingProxyType({n: LockedFile(sha(d), len(d)) for n, d in files.items()}))


def write_folder(folder: Path, files: dict[str, bytes] = FILES) -> Path:
    for name, data in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
    return folder


# ---------------------------------------------------------------- the committed lock


def test_the_committed_lock_pins_both_roles_by_full_commit_sha_with_sizes():
    lock = ModelLock.load(REAL_MODELS_LOCK)
    assert set(lock.models) == {"embedder", "reranker"}
    assert lock["embedder"].repo == "BAAI/bge-small-en-v1.5"
    assert lock["reranker"].repo == "cross-encoder/ms-marco-MiniLM-L6-v2"
    for spec in lock.models.values():
        assert len(spec.revision) == 40
        assert spec.files["model.safetensors"].bytes > 50_000_000
        assert all(name.endswith((".safetensors", ".json", ".txt")) for name in spec.files)


def test_a_lock_survives_a_save_and_load(tmp_path):
    lock = ModelLock.load(REAL_MODELS_LOCK)
    lock.save(tmp_path / "models.lock.json")
    again = ModelLock.load(tmp_path / "models.lock.json")
    assert {r: s.fingerprint for r, s in again.models.items()} == {r: s.fingerprint for r, s in lock.models.items()}
    assert (tmp_path / "models.lock.json").read_text() == REAL_MODELS_LOCK.read_text()


def test_the_fingerprint_moves_with_the_revision_every_hash_and_every_size():
    spec = spec_for({"model.safetensors": b"a"})
    files = spec.files
    assert spec.fingerprint != ModelSpec("embedder", "owner/model", "f" * 40, "MIT", files).fingerprint
    assert spec.fingerprint != spec_for({"model.safetensors": b"b"}).fingerprint
    assert spec.fingerprint != ModelSpec("embedder", "other/model", REVISION, "MIT", files).fingerprint
    resized = {"model.safetensors": LockedFile(files["model.safetensors"].sha256, 99)}
    assert spec.fingerprint != ModelSpec("embedder", "owner/model", REVISION, "MIT", resized).fingerprint


# ---------------------------------------------------------------- what a lock may say


@pytest.mark.parametrize("bad, message", [
    (entry(revision="main"), "full 40-character commit SHA"),
    (entry(revision=REVISION[:12]), "full 40-character commit SHA"),
    (entry(files={"pytorch_model.bin": locked(b"x"), "model.safetensors": locked(b"y")}), "not an allowed file type"),
    (entry(files={"model.safetensors": locked(b"y"), "modeling_custom.py": locked(b"z")}), "not an allowed file type"),
    (entry(files={"model.safetensors": locked(b"y"), "onnx/model.onnx": locked(b"z")}), "not an allowed file type"),
    (entry(files={"../escape.json": locked(b"x"), "model.safetensors": locked(b"y")}), "plain relative path"),
    (entry(files={"/abs/config.json": locked(b"x"), "model.safetensors": locked(b"y")}), "plain relative path"),
    (entry(files={"config.json": locked(b"x")}), "no safetensors weights"),
    (entry(files={"model.safetensors": {"sha256": "abc", "bytes": 1}}), "64-character lowercase SHA-256"),
    (entry(files={"model.safetensors": {"sha256": sha(b"x").upper(), "bytes": 1}}), "64-character lowercase"),
    (entry(files={"model.safetensors": {"sha256": sha(b"x"), "bytes": -1}}), "size in bytes"),
    (entry(files={"model.safetensors": {"sha256": sha(b"x"), "bytes": True}}), "size in bytes"),
    (entry(files={"model.safetensors": sha(b"x")}), "exactly sha256 and bytes"),
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


def test_a_matching_copy_has_no_problems_and_every_difference_is_named(tmp_path):
    spec = spec_for()
    folder = write_folder(tmp_path / "m")
    assert problems(spec, folder) == []
    (folder / "config.json").write_bytes(b'{"changed": 1}')
    (folder / "1_Pooling" / "config.json").unlink()
    (folder / "pytorch_model.bin").write_bytes(b"pickle")  # never locked, so never trusted
    assert problems(spec, folder) == [
        "1_Pooling/config.json: missing",
        "config.json: differs from models.lock.json",
        "pytorch_model.bin: not in models.lock.json",
    ]


def test_a_same_size_edit_is_still_caught(tmp_path):
    folder = write_folder(tmp_path / "m")
    (folder / "model.safetensors").write_bytes(b"WEIGHTS")  # same length, different bytes
    assert problems(spec_for(), folder) == ["model.safetensors: differs from models.lock.json"]


def test_require_names_the_fix_for_a_missing_or_altered_copy(tmp_path):
    spec = spec_for()
    with pytest.raises(ModelError, match="models fetch"):
        require(spec, tmp_path / "models")
    folder = write_folder(tmp_path / "models" / spec.folder_name)
    assert require(spec, tmp_path / "models") == folder
    (folder / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ModelError, match="doesn't match models.lock.json: model.safetensors"):
        require(spec, tmp_path / "models")


def test_fetch_copies_exactly_the_locked_files_from_the_hub_and_checks_them(tmp_path, monkeypatch):
    hub = write_folder(tmp_path / "hub", {**FILES, "pytorch_model.bin": b"never asked for"})
    asked = []

    def fake_download(repo, filename, revision):
        asked.append((repo, filename, revision))
        return hub / filename

    monkeypatch.setattr(models, "_download", fake_download)
    spec = spec_for()
    folder = fetch(spec, tmp_path / "models")
    assert sorted(f for _, f, _ in asked) == sorted(FILES) and {r for _, _, r in asked} == {REVISION}
    assert problems(spec, folder) == [] and not (folder / "pytorch_model.bin").exists()
    (hub / "config.json").write_bytes(b"{ swapped upstream }")
    (folder / "config.json").unlink()
    with pytest.raises(ModelError, match="config.json doesn't match models.lock.json"):
        fetch(spec, tmp_path / "models")
    assert not list(folder.glob("*.tmp-*"))


def test_nothing_large_is_kept_where_icloud_syncs_it(tmp_path):
    synced = tmp_path / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "proj" / "data" / "cache"
    synced.mkdir(parents=True)
    with pytest.raises(ModelError, match="inside iCloud Drive"):
        ensure_not_synced(synced, "cache")
    skipped = tmp_path / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "proj" / "data.nosync" / "cache"
    skipped.mkdir(parents=True)
    ensure_not_synced(skipped, "cache")
    ensure_not_synced(tmp_path, "cache")


def test_pin_refuses_a_branch_name():
    with pytest.raises(ModelError, match="full 40-character commit SHA"):
        models.pin("embedder", "owner/model", "main", "MIT")


# ---------------------------------------------------------------- loaded weights


def test_loaded_weights_must_equal_the_file_tensor_for_tensor(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from raglab.neural import check_weights

    module = torch.nn.Linear(3, 2)
    weights = tmp_path / "model.safetensors"
    stored = {k: v.clone() for k, v in module.state_dict().items()}
    save_file({**stored, "embeddings.position_ids": torch.arange(4)}, str(weights))  # a buffer the file may carry
    check_weights(module, weights)
    with torch.no_grad():
        module.weight[0, 0] += 1e-6  # a randomly initialised or altered layer
    with pytest.raises(ModelError, match="different \\['weight'\\]"):
        check_weights(module, weights)
    save_file({"weight": stored["weight"]}, str(weights))
    with pytest.raises(ModelError, match="missing \\['bias'\\]"):
        check_weights(torch.nn.Linear(3, 2), weights)


# ---------------------------------------------------------------- the CLI


def test_models_verify_reports_each_role(tmp_path, monkeypatch):
    spec = spec_for()
    lock_path = tmp_path / "models.lock.json"
    reranker = ModelSpec("reranker", "owner/rr", REVISION, "MIT", spec.files)
    ModelLock(MappingProxyType({"embedder": spec, "reranker": reranker})).save(lock_path)
    write_folder(tmp_path / "models" / spec.folder_name)
    monkeypatch.setattr("raglab.cli.default_models_lock_path", lambda: lock_path)
    monkeypatch.setattr("raglab.models.default_models_dir", lambda: tmp_path / "models")
    result = CliRunner().invoke(app, ["models", "verify"])
    assert result.exit_code == 1
    assert "OK embedder" in result.output and "not downloaded" in result.output and "models fetch" in result.output
    write_folder(tmp_path / "models" / reranker.folder_name)
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
def test_the_real_embedder_gives_the_known_answer_and_is_deterministic_per_text(real_models):
    from raglab.neural import QUERY_INSTRUCTION, SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(ModelLock.load(REAL_MODELS_LOCK)["embedder"])
    doc = "FastAPI validates the request body with a Pydantic model."
    vectors = embedder.embed([doc, "Mount StaticFiles to serve images from a directory.", "x"], "")
    alone = np.stack([embedder.embed([t], "")[0] for t in (doc, "Mount StaticFiles to serve images from a directory.", "x")])
    assert vectors.shape == (3, 384) and vectors.dtype == np.float32
    assert np.array_equal(vectors, alone)  # a vector never depends on its batch
    # Known answers, recorded on the author's machine: a wrongly loaded model would miss them.
    assert np.allclose(vectors[0, :6], [-0.098518, 0.034436, -0.032952, -0.052906, 0.023855, -0.025706], atol=1e-5)
    query = embedder.embed(["How is a request body validated?"], QUERY_INSTRUCTION)[0]
    assert np.allclose(query[:6], [-0.055021, -0.017972, -0.013198, -0.078792, 0.020649, -0.036077], atol=1e-5)
    assert float(vectors[0].astype(np.float64) @ query.astype(np.float64)) == pytest.approx(0.765583, abs=1e-5)
    assert not np.array_equal(query, embedder.embed(["How is a request body validated?"], "")[0])
    assert embedder.settings["threads"] == 4 and embedder.settings["batch_size"] == 1


@pytest.mark.models
def test_the_real_reranker_gives_the_known_answer(real_models):
    from raglab.neural import CrossEncoderScorer

    scorer = CrossEncoderScorer(ModelLock.load(REAL_MODELS_LOCK)["reranker"])
    passages = ["FastAPI validates the request body with a Pydantic model.",
                "Mount StaticFiles to serve images from a directory."]
    scores = scorer.score("How is a request body validated?", passages)
    assert np.allclose(scores, [5.25845, -11.46333], atol=1e-4)  # raw logits: no sigmoid
    assert np.array_equal(scores, scorer.score("How is a request body validated?", passages))
