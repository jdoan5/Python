"""The two pretrained models Stage 3 adds, pinned and checked the way the corpus is.

`models.lock.json` sits beside `corpus.lock.json` at the project root. For each
role, the embedder and the reranker, it names:

- the Hugging Face repository;
- the exact revision (a commit SHA, never a branch name);
- the SHA-256 of every file the model is loaded from.

The locked files are only weights in safetensors format, configuration and
tokenizer files. There are no pickled weights and no Python files, and models
are loaded with `trust_remote_code=False` and `use_safetensors=True`. So
loading a model can't run code that came with the download.

`raglab models fetch` downloads exactly the locked files into the Hugging Face
cache and checks them. A run loads each model from that local copy, never the
network, and checks every hash first, so it can't silently use different
weights.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

from raglab.corpus import project_root

LOCK_SCHEMA = 1
ROLES = ("embedder", "reranker")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
# What a lock may name: weights only as safetensors, plus configuration and tokenizer files.
# Pickled weights (.bin, .pt, ...) can run code when loaded, and .py files are remote code.
_ALLOWED_SUFFIXES = (".safetensors", ".json", ".txt")
# What `pin` takes from a repository when a lock is first written.
PIN_PATTERNS = ("*.json", "*.txt", "model.safetensors", "1_Pooling/*")
_PIN_EXCLUDED = ("README.md",)


class ModelError(Exception):
    """A model is missing, unpinned, or does not match its lock."""


@dataclass(frozen=True)
class ModelSpec:
    role: str
    repo: str
    revision: str
    license: str
    files: Mapping[str, str]  # path inside the repository -> SHA-256

    @property
    def fingerprint(self) -> str:
        """One hash over the repository, the revision and every locked file."""
        h = hashlib.sha256(f"{self.repo}\0{self.revision}\0".encode("utf-8"))
        for name, digest in sorted(self.files.items()):
            h.update(f"{name}\0{digest}\0".encode("utf-8"))
        return h.hexdigest()

    def manifest(self) -> dict:
        return {
            "repo": self.repo, "revision": self.revision, "license": self.license,
            "files": len(self.files), "fingerprint": self.fingerprint,
        }

    def as_json(self) -> dict:
        return {"repo": self.repo, "revision": self.revision, "license": self.license,
                "files": dict(sorted(self.files.items()))}


@dataclass(frozen=True)
class ModelLock:
    models: Mapping[str, ModelSpec]

    def __getitem__(self, role: str) -> ModelSpec:
        try:
            return self.models[role]
        except KeyError:
            raise ModelError(f"models.lock.json pins no {role!r} model; it pins {sorted(self.models)}") from None

    def save(self, path: Path) -> None:
        data = {"schema": LOCK_SCHEMA, "models": {r: self.models[r].as_json() for r in ROLES if r in self.models}}
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ModelLock":
        if not path.is_file():
            raise ModelError(f"No model lock at {path.name}. Restore models.lock.json from git.")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ModelError(f"{path.name} is unreadable or not valid JSON ({e}). Restore it from git.") from e
        if not isinstance(data, dict) or data.get("schema") != LOCK_SCHEMA or not isinstance(data.get("models"), dict):
            raise ModelError(f"{path.name} is not a schema-{LOCK_SCHEMA} model lock. Restore it from git.")
        if set(data) != {"schema", "models"}:
            raise ModelError(f"{path.name} has unexpected keys {sorted(set(data) - {'schema', 'models'})}.")
        models = {}
        for role, entry in data["models"].items():
            models[role] = _parse_spec(path.name, role, entry)
        return cls(MappingProxyType(models))


def _parse_spec(lock_name: str, role: str, entry: object) -> ModelSpec:
    where = f"{lock_name}: model {role!r}"
    if role not in ROLES:
        raise ModelError(f"{where} is not a known role; roles are {list(ROLES)}")
    if not isinstance(entry, dict) or set(entry) != {"repo", "revision", "license", "files"}:
        raise ModelError(f"{where} must have exactly repo, revision, license and files")
    repo, revision, license_, files = entry["repo"], entry["revision"], entry["license"], entry["files"]
    if not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ModelError(f"{where}: repo must look like 'owner/name'")
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ModelError(f"{where}: revision must be a full 40-character commit SHA, not a branch or tag")
    if not isinstance(license_, str) or not license_:
        raise ModelError(f"{where}: license must be named")
    if not isinstance(files, dict) or not files:
        raise ModelError(f"{where}: files must map file names to SHA-256 digests")
    for name, digest in files.items():
        _check_file_name(where, name)
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ModelError(f"{where}: {name!r} needs a 64-character lowercase SHA-256")
    if not any(name.endswith(".safetensors") for name in files):
        raise ModelError(f"{where}: no safetensors weights are locked")
    return ModelSpec(role, repo, revision, license_, MappingProxyType(dict(files)))


def _check_file_name(where: str, name: object) -> None:
    if not isinstance(name, str) or not name:
        raise ModelError(f"{where}: file names must be non-empty strings")
    parts = PurePosixPath(name).parts
    if name.startswith("/") or "\\" in name or ".." in parts or name != PurePosixPath(name).as_posix():
        raise ModelError(f"{where}: {name!r} must be a plain relative path inside the repository")
    if not name.endswith(_ALLOWED_SUFFIXES):
        raise ModelError(f"{where}: {name!r} is not an allowed file type (safetensors weights, json, txt); "
                         "pickled weights and code are never loaded")


def default_models_lock_path() -> Path:
    return project_root() / "models.lock.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def problems(spec: ModelSpec, directory: Path) -> list[str]:
    """Every locked file that is missing from `directory` or differs from its hash."""
    found = []
    for name, digest in sorted(spec.files.items()):
        path = directory / name
        if not path.is_file():
            found.append(f"{name}: missing")
        elif _sha256(path) != digest:
            found.append(f"{name}: SHA-256 differs from models.lock.json")
    return found


def local_dir(spec: ModelSpec) -> Path | None:
    """Where the locked revision sits in the Hugging Face cache, or None if it was never fetched."""
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(spec.repo, revision=spec.revision, allow_patterns=sorted(spec.files),
                                 local_files_only=True)
    except Exception:  # LocalEntryNotFoundError and friends: nothing usable in the cache
        return None
    return Path(path)


def require(spec: ModelSpec) -> Path:
    """The verified local copy of a locked model, or a ModelError that says how to get it."""
    directory = local_dir(spec)
    if directory is None:
        raise ModelError(f"The {spec.role} {spec.repo} isn't downloaded. Run `python -m raglab models fetch`.")
    found = problems(spec, directory)
    if found:
        raise ModelError(f"The local copy of {spec.repo} doesn't match models.lock.json: {'; '.join(found)}. "
                         "Delete it from the Hugging Face cache and run `python -m raglab models fetch`.")
    return directory


def fetch(spec: ModelSpec) -> Path:
    """Download exactly the locked files at the locked revision, then check every hash."""
    from huggingface_hub import snapshot_download

    directory = Path(snapshot_download(spec.repo, revision=spec.revision, allow_patterns=sorted(spec.files)))
    found = problems(spec, directory)
    if found:
        raise ModelError(f"{spec.repo} at {spec.revision[:12]} doesn't match models.lock.json: {'; '.join(found)}")
    return directory


def pin(role: str, repo: str, revision: str, license_: str) -> ModelSpec:
    """Download a revision's configuration, tokenizer and safetensors weights, and lock their hashes.

    Used once to write models.lock.json. Weights must already be in safetensors format.
    """
    from huggingface_hub import snapshot_download

    if not _REVISION_RE.fullmatch(revision):
        raise ModelError("pin a full 40-character commit SHA, not a branch or tag")
    directory = Path(snapshot_download(repo, revision=revision, allow_patterns=list(PIN_PATTERNS)))
    files = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        name = path.relative_to(directory).as_posix()
        if name in _PIN_EXCLUDED:
            continue
        _check_file_name(f"pin {repo}", name)
        files[name] = _sha256(path)
    return _parse_spec("pin", role, {"repo": repo, "revision": revision, "license": license_, "files": files})
