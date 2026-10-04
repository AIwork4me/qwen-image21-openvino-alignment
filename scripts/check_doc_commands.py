#!/usr/bin/env python3
"""Sanity-check every shell command embedded in project markdown docs.

Extracts fenced-code-block commands from the given markdown files and verifies
that every command that references a repository file resolves to a committed
path. Usage:

    python scripts/check_doc_commands.py [file.md ...]

Defaults to the v1+v2 document set. Exit 1 on any unresolved reference.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DOCS = [
    "README.md",
    "CONCLUSIONS.md",
    "reports/REPRODUCTION.md",
    "reports/CUSTOMER_REPORT.md",
    "reports/TECHNICAL_REPORT.md",
    "reports/EXECUTIVE_SUMMARY.md",
    "reports/LIMITATIONS.md",
]

SKIP_PREFIXES = ("#", "PY")  # comment / heredoc body (blank handled separately)


def extract_commands(md_path):
    """Yield (line_no, command) for shell lines inside ``` fences."""
    lines = open(md_path, encoding="utf-8").read().splitlines()
    in_fence = False
    fence_tag = ""
    for i, ln in enumerate(lines, 1):
        stripped = ln.strip()
        if stripped.startswith("```"):
            if not in_fence:
                fence_tag = stripped[3:].strip().lower()
            in_fence = not in_fence
            continue
        if in_fence and fence_tag in ("", "bash", "sh", "shell", "console"):
            if stripped.startswith("$ "):
                stripped = stripped[2:]
            if not stripped or stripped.startswith(SKIP_PREFIXES):
                continue
            yield i, stripped


def referenced_repo_paths(cmd):
    """Return repo-relative paths the command text references, if checkable."""
    # strip common wrappers and env assignments
    m = re.match(r"^(?:sudo\s+)?(?:python(?:3)?|-m pip|pip3?|git|bash|sh|curl|wget|tar|ln|mkdir|cp|mv|cat|du|ls|cd|echo)\b", cmd)
    if not m:
        return []
    paths = set()
    for tok in re.findall(r"[\w./-]+", cmd):
        if not ("/" in tok or tok.endswith((".py", ".sh", ".yaml", ".json", ".md"))):
            continue
        if tok.startswith(("http", "https", "PY", "main", "-")):
            continue
        # scripts/..., config/..., tests/..., prompts/... references
        for prefix in ("scripts/", "config/", "tests/", "prompts/", "reports/", "docs/", "artifacts/v2/"):
            if prefix in tok:
                p = tok[tok.index(prefix):].rstrip('",:;)')
                p = p.split()[0] if p.split() else p
                paths.add(p)
    return sorted(paths)


def main():
    docs = sys.argv[1:] or [os.path.join(ROOT, d) for d in DEFAULT_DOCS if os.path.exists(os.path.join(ROOT, d))]
    failures = 0
    checked = 0
    for doc in docs:
        for line_no, cmd in extract_commands(doc):
            checked += 1
            for p in referenced_repo_paths(cmd):
                if not os.path.exists(os.path.join(ROOT, p)):
                    print(f"MISSING {doc}:{line_no}: {p}  (cmd: {cmd[:100]})")
                    failures += 1
        # direct "bash X" / "python X" file references
        for line_no, cmd in extract_commands(doc):
            m = re.match(r"^(?:bash|sh|python3?)\s+([\w./-]+)", cmd)
            if m:
                target = m.group(1)
                if target.startswith("http"):
                    continue
                if target.endswith((".py", ".sh")) and not os.path.exists(os.path.join(ROOT, target)):
                    print(f"MISSING-TARGET {doc}:{line_no}: {target}")
                    failures += 1
                # wrong interpreter: bash/sh on a .py file
                if re.match(r"^(?:bash|sh)\s+[\w./-]+\.py\b", cmd):
                    print(f"WRONG-INTERPRETER {doc}:{line_no}: {cmd[:100]}")
                    failures += 1
    print(f"checked {checked} fenced shell commands across {len(docs)} docs: "
          f"{'PASS' if failures == 0 else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
