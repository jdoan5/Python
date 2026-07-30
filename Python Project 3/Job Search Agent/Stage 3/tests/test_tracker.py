"""Tracker tests — stdlib module, no API key or streamlit needed.

Run from the Stage 3 folder:  ../.venv/bin/python -m pytest -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracker import (  # noqa: E402
    TrackerError,
    gap_frequencies,
    load_applications,
    parse_sidecar,
    sync_output_dir,
    update_application,
)


def write_sidecar(directory: Path, name: str, company: str = "Acme",
                  fit: int = 7, gaps=None) -> Path:
    payload = {
        "source": "https://example.com/job",
        "requirements": {"company": company, "role": "Backend Engineer",
                         "seniority": "mid"},
        "draft": {
            "fit_score": fit,
            "bullets": [{"text": "b1"}, {"text": "b2"}],
            "gaps": gaps if gaps is not None else ["Kubernetes experience"],
        },
    }
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_sidecar_extracts_fields(tmp_path: Path) -> None:
    sidecar = write_sidecar(tmp_path, "a.json")
    parsed = parse_sidecar(sidecar)
    assert parsed is not None
    assert parsed["company"] == "Acme"
    assert parsed["fit_score"] == 7
    assert parsed["bullet_count"] == 2
    assert json.loads(parsed["gaps"]) == ["Kubernetes experience"]


def test_parse_sidecar_malformed_returns_none(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert parse_sidecar(bad) is None
    not_dict = tmp_path / "list.json"
    not_dict.write_text("[1,2]", encoding="utf-8")
    assert parse_sidecar(not_dict) is None


def test_sync_is_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "output"
    out.mkdir()
    db = tmp_path / "t.db"
    write_sidecar(out, "one.json")
    write_sidecar(out, "two.json", company="Beta")

    added, skipped = sync_output_dir(db_path=db, output_dir=out)
    assert added == 2 and skipped == []
    added2, _ = sync_output_dir(db_path=db, output_dir=out)
    assert added2 == 0
    assert len(load_applications(db_path=db)) == 2


def test_sync_preserves_user_edits(tmp_path: Path) -> None:
    out = tmp_path / "output"
    out.mkdir()
    db = tmp_path / "t.db"
    write_sidecar(out, "one.json")
    sync_output_dir(db_path=db, output_dir=out)
    app = load_applications(db_path=db)[0]
    update_application(app["id"], status="interviewing", notes="phone screen Fri",
                       db_path=db)
    # Re-sync must not clobber the user's status/notes.
    sync_output_dir(db_path=db, output_dir=out)
    app2 = load_applications(db_path=db)[0]
    assert app2["status"] == "interviewing"
    assert app2["notes"] == "phone screen Fri"


def test_sync_skips_malformed_and_counts_it(tmp_path: Path) -> None:
    out = tmp_path / "output"
    out.mkdir()
    db = tmp_path / "t.db"
    write_sidecar(out, "good.json")
    (out / "broken.json").write_text("{oops", encoding="utf-8")
    added, skipped = sync_output_dir(db_path=db, output_dir=out)
    assert added == 1
    assert skipped == ["broken.json"]


def test_sync_missing_dir_is_noop(tmp_path: Path) -> None:
    added, skipped = sync_output_dir(db_path=tmp_path / "t.db",
                                     output_dir=tmp_path / "nope")
    assert (added, skipped) == (0, [])


def test_update_rejects_bad_status(tmp_path: Path) -> None:
    out = tmp_path / "output"
    out.mkdir()
    db = tmp_path / "t.db"
    write_sidecar(out, "one.json")
    sync_output_dir(db_path=db, output_dir=out)
    app = load_applications(db_path=db)[0]
    with pytest.raises(TrackerError, match="Invalid status"):
        update_application(app["id"], status="ghosted", db_path=db)


def test_update_unknown_id_raises(tmp_path: Path) -> None:
    with pytest.raises(TrackerError, match="No application"):
        update_application(999, status="applied", db_path=tmp_path / "t.db")


def test_gap_frequencies_normalizes_and_ranks() -> None:
    apps = [
        {"gaps": ["Kubernetes experience", "Terraform"]},
        {"gaps": ["kubernetes   experience"]},   # same gap, different case/spacing
        {"gaps": ["Terraform"]},
        {"gaps": []},
    ]
    ranked = gap_frequencies(apps)
    assert ranked[0] == ("kubernetes experience", 2)
    assert ("terraform", 2) in ranked


def test_gap_frequencies_dedupes_within_one_application() -> None:
    # Count = "N postings asked for this", so a duplicate inside one
    # application must count once.
    apps = [{"gaps": ["Terraform", "terraform", "TERRAFORM"]}]
    assert gap_frequencies(apps) == [("terraform", 1)]


def test_parse_sidecar_rejects_foreign_json(tmp_path: Path) -> None:
    # A stray JSON dict (config file, other tool's output) must not become
    # an "Unknown/Unknown" application.
    stray = tmp_path / "package.json"
    stray.write_text(json.dumps({"name": "web-app", "version": "1.0"}), encoding="utf-8")
    assert parse_sidecar(stray) is None


def test_parse_sidecar_fit_score_edge_cases(tmp_path: Path) -> None:
    # NaN must not crash the sync; bools are not scores; floats round.
    nan_file = tmp_path / "nan.json"
    nan_file.write_text(
        '{"requirements": {}, "draft": {"fit_score": NaN}}', encoding="utf-8"
    )
    assert parse_sidecar(nan_file)["fit_score"] is None

    bool_file = tmp_path / "bool.json"
    bool_file.write_text(
        '{"requirements": {}, "draft": {"fit_score": true}}', encoding="utf-8"
    )
    assert parse_sidecar(bool_file)["fit_score"] is None

    float_file = tmp_path / "float.json"
    float_file.write_text(
        '{"requirements": {}, "draft": {"fit_score": 7.5}}', encoding="utf-8"
    )
    assert parse_sidecar(float_file)["fit_score"] == 8  # round, not truncate


def test_created_at_parsed_from_filename_stamp(tmp_path: Path) -> None:
    # The save_draft filename stamp survives copies; mtime does not.
    sidecar = write_sidecar(tmp_path, "2026-07-29_101530_123456_Acme.json")
    parsed = parse_sidecar(sidecar)
    assert parsed["created_at"] == "2026-07-29T10:15:30"
    # Pre-microsecond-era filenames still parse.
    old = write_sidecar(tmp_path, "2026-01-02_030405_Beta.json", company="Beta")
    assert parse_sidecar(old)["created_at"] == "2026-01-02T03:04:05"
