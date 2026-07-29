"""Pipeline tests that need no API key: grounding validation, evidence pack,
saving, and the HITL gate contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from job_agent.pipeline import (
    EvidencePack,
    GroundingError,
    _validate_grounding,
    save_draft,
)
from job_agent.retrieval import InventoryEntry, SearchHit
from job_agent.schemas import (
    EvidenceMatch,
    JobRequirements,
    Requirement,
    TailoredBullet,
    TailoredDraft,
)


def make_evidence() -> EvidencePack:
    pack = EvidencePack()
    pack.add_hits(
        "python api",
        [SearchHit(InventoryEntry("py-api", "project", "FastAPI service."), 2.0)],
    )
    return pack


def make_draft(evidence_ids: list) -> TailoredDraft:
    return TailoredDraft(
        fit_score=7,
        fit_rationale="Solid overlap on Python and APIs.",
        matches=[EvidenceMatch(requirement="Python", evidence_ids=evidence_ids,
                               strength="strong", note="ok")],
        bullets=[TailoredBullet(text="Built a FastAPI service.",
                                evidence_ids=evidence_ids,
                                targets_requirement="Python APIs")],
        gaps=[],
        cover_note=None,
    )


def make_requirements() -> JobRequirements:
    return JobRequirements(
        company="Acme", role="Backend Engineer", seniority="mid",
        requirements=[Requirement(text="Python", kind="must_have", keywords=["python"])],
        red_flags=[],
    )


def test_grounding_passes_with_known_ids() -> None:
    _validate_grounding(make_draft(["py-api"]), make_evidence())


def test_grounding_rejects_unknown_ids() -> None:
    with pytest.raises(GroundingError, match="never retrieved"):
        _validate_grounding(make_draft(["fabricated-entry"]), make_evidence())


def test_grounding_rejects_empty_evidence() -> None:
    with pytest.raises(GroundingError, match="no evidence"):
        _validate_grounding(make_draft([]), make_evidence())


def test_evidence_pack_dedupes_entries() -> None:
    pack = EvidencePack()
    entry = InventoryEntry("x", "skill", "SQL.")
    pack.add_hits("sql", [SearchHit(entry, 1.0)])
    pack.add_hits("database", [SearchHit(entry, 0.8)])
    assert len(pack.entries) == 1
    assert pack.searches == ["sql", "database"]


def test_evidence_pack_prompt_block_empty() -> None:
    assert "no evidence" in EvidencePack().as_prompt_block()


def test_save_draft_writes_md_and_json(tmp_path: Path) -> None:
    md_path = save_draft(
        tmp_path, make_requirements(), make_evidence(), make_draft(["py-api"]),
        source="https://example.com/job",
    )
    assert md_path.exists()
    assert md_path.suffix == ".md"
    sidecar = md_path.with_suffix(".json")
    assert sidecar.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "Backend Engineer" in content
    assert "Fit: 7/10" in content
    assert "py-api" in content


def test_save_draft_sanitizes_company_name(tmp_path: Path) -> None:
    reqs = make_requirements()
    reqs.company = "Evil/Corp: <Ltd>"
    md_path = save_draft(tmp_path, reqs, make_evidence(), make_draft(["py-api"]), "src")
    assert "/" not in md_path.name.replace(md_path.suffix, "")
    assert md_path.parent == tmp_path


def test_save_draft_same_second_never_overwrites(tmp_path: Path) -> None:
    args = (tmp_path, make_requirements(), make_evidence(), make_draft(["py-api"]), "src")
    first = save_draft(*args)
    second = save_draft(*args)
    assert first != second
    assert first.exists() and second.exists()


def test_grounding_rejects_unknown_match_ids() -> None:
    draft = make_draft(["py-api"])
    draft.matches[0].evidence_ids = ["fabricated-id"]
    with pytest.raises(GroundingError, match="Match cites"):
        _validate_grounding(draft, make_evidence())


def test_grounding_allows_empty_match_ids() -> None:
    # strength='none' matches legitimately cite nothing.
    draft = make_draft(["py-api"])
    draft.matches[0].evidence_ids = []
    draft.matches[0].strength = "none"
    _validate_grounding(draft, make_evidence())
