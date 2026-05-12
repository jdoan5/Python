from __future__ import annotations

from pathlib import Path

import pytest

from apps.github_triage.tools import SafePath, build_tools


def test_safe_path_blocks_traversal(tmp_path: Path) -> None:
    safe = SafePath(tmp_path)
    with pytest.raises(ValueError):
        safe.resolve("../../etc/passwd")


def test_safe_path_resolves_inside_repo(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("hello")
    safe = SafePath(tmp_path)
    resolved = safe.resolve("src/a.py")
    assert resolved.is_file()


def test_search_codebase_finds_match(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("def load_config():\n    return {}\n")
    tools = build_tools(tmp_path)
    result, is_error = tools.execute("search_codebase", {"query": "load_config"})
    assert not is_error
    assert "module.py" in result
    assert "def load_config" in result


def test_read_file_returns_numbered_lines(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("a\nb\nc\n")
    tools = build_tools(tmp_path)
    result, is_error = tools.execute("read_file", {"path": "x.py"})
    assert not is_error
    assert "    1: a" in result
    assert "    2: b" in result


def test_list_files_excludes_ignored(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("")
    tools = build_tools(tmp_path)
    result, _ = tools.execute("list_files", {})
    assert "main.py" in result
    assert ".git" not in result


def test_read_file_rejects_traversal(tmp_path: Path) -> None:
    tools = build_tools(tmp_path)
    result, is_error = tools.execute("read_file", {"path": "../secret"})
    assert is_error
