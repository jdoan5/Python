from __future__ import annotations

from pathlib import Path

from apps.job_copilot.tools import _html_to_text, build_tools


def test_html_to_text_strips_tags() -> None:
    html_input = "<html><body><h1>Senior Engineer</h1><p>Build &amp; ship.</p></body></html>"
    result = _html_to_text(html_input)
    assert "Senior Engineer" in result
    assert "Build & ship" in result
    assert "<h1>" not in result
    assert "&amp;" not in result


def test_html_to_text_removes_scripts() -> None:
    html_input = "<p>visible</p><script>alert('x')</script><style>.a{}</style>"
    result = _html_to_text(html_input)
    assert "visible" in result
    assert "alert" not in result
    assert ".a{}" not in result


def test_save_and_list_applications(tmp_path: Path) -> None:
    resume = tmp_path / "resume.txt"
    resume.write_text("Senior backend engineer with 5 years of Python.")
    db = tmp_path / "apps.db"
    tools = build_tools(resume, db)

    out, err = tools.execute(
        "save_application",
        {
            "company": "Acme",
            "role": "Backend Engineer",
            "fit_score": 8,
            "url": "https://example.com/job",
            "notes": "Strong Python match.",
        },
    )
    assert not err
    assert "Acme" in out

    out, err = tools.execute("list_applications", {})
    assert not err
    assert "Acme" in out and "Backend Engineer" in out


def test_save_application_rejects_invalid_score(tmp_path: Path) -> None:
    resume = tmp_path / "resume.txt"
    resume.write_text("x")
    db = tmp_path / "apps.db"
    tools = build_tools(resume, db)
    out, err = tools.execute(
        "save_application",
        {"company": "X", "role": "Y", "fit_score": 99},
    )
    assert err
    assert "1-10" in out


def test_fetch_url_rejects_non_http(tmp_path: Path) -> None:
    resume = tmp_path / "resume.txt"
    resume.write_text("x")
    db = tmp_path / "apps.db"
    tools = build_tools(resume, db)
    out, _ = tools.execute("fetch_url", {"url": "file:///etc/passwd"})
    assert "Refusing" in out


def test_read_resume_missing(tmp_path: Path) -> None:
    tools = build_tools(tmp_path / "nope.txt", tmp_path / "apps.db")
    out, _ = tools.execute("read_resume", {})
    assert "not found" in out.lower()
