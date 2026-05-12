from __future__ import annotations

from pathlib import Path

from agent_core.evals import EvalCase, run_evals
from apps.github_triage.agent import build_agent

CASES = [
    EvalCase(
        name="error_message_grounding",
        input=(
            "Title: ImportError when loading config\n\n"
            "I get `ImportError: cannot import name 'load_config' from 'agent_core.llm'` "
            "when I run the CLI. Stack trace points at apps/github_triage/cli.py."
        ),
        rubric=(
            "Output should: (1) cite at least one specific file:line reference, "
            "(2) classify the issue (bug/question/etc), (3) state whether 'load_config' "
            "actually exists in agent_core/llm.py based on reading the code, "
            "(4) be addressed to the reporter, not just internal reasoning."
        ),
    ),
    EvalCase(
        name="feature_vs_bug",
        input=(
            "Title: Support GitLab issues too\n\n"
            "Would be great if this tool could also triage GitLab issues, not just GitHub. "
            "We use GitLab Enterprise at work."
        ),
        rubric=(
            "Output should: (1) classify this as 'feature', not 'bug', "
            "(2) acknowledge the request thoughtfully without committing to implementing it, "
            "(3) NOT fabricate references to GitLab-related code (it doesn't exist)."
        ),
    ),
    EvalCase(
        name="needs_info",
        input=(
            "Title: doesn't work\n\nIt's broken. please fix"
        ),
        rubric=(
            "Output should: (1) classify as 'needs-info', "
            "(2) ask 2-3 targeted clarifying questions (what command, error message, env), "
            "(3) NOT make up details that weren't provided."
        ),
    ),
]


def main(repo: Path) -> None:
    agent = build_agent(repo_root=repo, verbose=False)
    results = run_evals(CASES, lambda inp: agent.run(inp))

    total = sum(r.score for r in results)
    max_score = 5 * len(results)
    print(f"\nResults: {total}/{max_score}\n")
    for r in results:
        print(f"  [{r.score}/5] {r.case.name}")
        print(f"        {r.rationale}")
        print(f"        ({r.iterations} iters)\n")


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())
