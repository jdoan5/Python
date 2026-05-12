from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from agent_core.tools import Tool, ToolRegistry

MAX_FILE_BYTES = 200_000
MAX_SEARCH_RESULTS = 30
MAX_DIR_ENTRIES = 100

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


class SafePath:
    def __init__(self, repo_root: Path) -> None:
        self.root = repo_root.resolve()
        if not self.root.exists():
            raise ValueError(f"Repo root does not exist: {self.root}")

    def resolve(self, rel_path: str) -> Path:
        candidate = (self.root / rel_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as e:
            raise ValueError(f"Path escapes repo root: {rel_path}") from e
        return candidate


def build_tools(repo_root: Path) -> ToolRegistry:
    safe = SafePath(repo_root)
    registry = ToolRegistry()

    def search_codebase(query: str, file_pattern: str = "*") -> str:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        matches = []
        for path in safe.root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in IGNORE_DIRS for part in path.parts):
                continue
            if not fnmatch.fnmatch(path.name, file_pattern):
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                content = path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue
            for lineno, line in enumerate(content.splitlines(), 1):
                if pattern.search(line):
                    rel = path.relative_to(safe.root)
                    matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(matches) >= MAX_SEARCH_RESULTS:
                        break
            if len(matches) >= MAX_SEARCH_RESULTS:
                break
        if not matches:
            return f"No matches for '{query}' in pattern '{file_pattern}'."
        return "\n".join(matches)

    registry.register(
        Tool(
            name="search_codebase",
            description=(
                "Search the codebase for occurrences of a literal string (case-insensitive). "
                "Returns up to 30 matches as file:line: content. Use file_pattern (glob) to "
                "scope the search, e.g. '*.py'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Literal string to search for."},
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional glob pattern, e.g. '*.py'. Defaults to '*'.",
                    },
                },
                "required": ["query"],
            },
            fn=search_codebase,
        )
    )

    def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
        target = safe.resolve(path)
        if not target.is_file():
            return f"Not a file: {path}"
        if target.stat().st_size > MAX_FILE_BYTES:
            return f"File too large ({target.stat().st_size} bytes); read a smaller slice."
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        sliced = lines[max(0, start_line - 1) : end_line]
        numbered = [f"{i:>5}: {line}" for i, line in enumerate(sliced, start=start_line)]
        return "\n".join(numbered) if numbered else "(empty range)"

    registry.register(
        Tool(
            name="read_file",
            description=(
                "Read a file from the repo by relative path. Returns numbered lines. "
                "Defaults to first 200 lines; specify start_line/end_line for a slice."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to repo root."},
                    "start_line": {"type": "integer", "description": "1-indexed start line."},
                    "end_line": {"type": "integer", "description": "1-indexed end line."},
                },
                "required": ["path"],
            },
            fn=read_file,
        )
    )

    def list_files(directory: str = ".") -> str:
        target = safe.resolve(directory)
        if not target.is_dir():
            return f"Not a directory: {directory}"
        entries = []
        for entry in sorted(target.iterdir()):
            if entry.name in IGNORE_DIRS or entry.name.startswith("."):
                continue
            kind = "dir" if entry.is_dir() else "file"
            entries.append(f"{kind:4} {entry.relative_to(safe.root)}")
            if len(entries) >= MAX_DIR_ENTRIES:
                entries.append(f"... (truncated at {MAX_DIR_ENTRIES})")
                break
        return "\n".join(entries) if entries else "(empty)"

    registry.register(
        Tool(
            name="list_files",
            description="List files/directories in a repo path. Defaults to repo root.",
            input_schema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Path relative to repo root. Defaults to '.'.",
                    }
                },
                "required": [],
            },
            fn=list_files,
        )
    )

    return registry
