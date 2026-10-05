#!/usr/bin/env python3
"""Snapshot-mode git push over the GitHub Git Data API.

Egress proxy on this host blocks CONNECT to github.com but allows
api.github.com, so `git push` cannot reach the remote.  This tool pushes the
exact tree of local HEAD as a new commit on the remote branch (fast-forward
only), diffing local files against the remote tree via the API so unchanged
blobs are never re-uploaded (git blob SHA-1s are content addressed and
comparable across local/remote).

Remote commits therefore mirror local commits 1:1 (same message/author) but
with different SHAs; provenance is preserved in the message trailer.

Usage:
    python scripts/push_via_gh_api.py [--repo OWNER/NAME] [--branch BRANCH]
                                      [--message MSG] [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gh(*args: str, stdin: str | None = None, retries: int = 3) -> dict | list:
    cmd = ["gh", "api"]
    if stdin is not None:
        cmd += ["--input", "-", "-H", "Content-Type: application/json"]
    cmd += list(args)
    last_err = ""
    for attempt in range(retries):
        proc = subprocess.run(cmd, capture_output=True, text=True, input=stdin, cwd=REPO_ROOT)
        if proc.returncode == 0:
            if not proc.stdout.strip():
                return {}
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                return {"raw": proc.stdout}
        last_err = f"rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
        if "502" in proc.stderr or "503" in proc.stderr or "timeout" in proc.stderr.lower():
            time.sleep(5 * (attempt + 1))
            continue
        break
    raise RuntimeError(f"gh api failed ({' '.join(args)}): {last_err}")


def git(*args: str, binary: bool = False, check: bool = True) -> str | bytes:
    proc = subprocess.run(["/usr/bin/git", *args], capture_output=True, cwd=REPO_ROOT)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace')}")
    return proc.stdout if binary else proc.stdout.decode(errors="replace").strip()


def remote_tree(repo: str, sha: str) -> tuple[dict[str, tuple[str, str]], str]:
    r = gh(f"repos/{repo}/git/trees/{sha}?recursive=1")
    return {e["path"]: (e["mode"], e["sha"]) for e in r.get("tree", []) if e["type"] == "blob"}, r["sha"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="AIwork4me/qwen-image21-openvino-alignment")
    ap.add_argument("--branch", default=None)
    ap.add_argument("--message", default=None, help="override commit message (default: local HEAD message)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    branch = args.branch or str(git("rev-parse", "--abbrev-ref", "HEAD"))
    head = str(git("rev-parse", "HEAD"))
    subject = str(git("log", "-1", "--format=%s"))
    body = str(git("log", "-1", "--format=%b"))
    author = json.loads(str(git("show", "-s", "--format=%an%x00%ae%x00%aI"))).split("\x00") if False else None

    ref = gh(f"repos/{args.repo}/git/ref/heads/{branch}")
    remote_sha = ref["object"]["sha"]
    print(f"local  HEAD {head[:10]}  ({branch})")
    print(f"remote tip {remote_sha[:10]}")

    rcommit = gh(f"repos/{args.repo}/git/commits/{remote_sha}")
    rfiles, rtree_sha = remote_tree(args.repo, rcommit["tree"]["sha"])
    print(f"remote tree: {len(rfiles)} files")

    local_out = str(git("ls-tree", "-r", "HEAD"))
    local_files: dict[str, tuple[str, str]] = {}
    for line in local_out.splitlines():
        meta, path = line.split("\t", 1)
        mode, _otype, osha = meta.split()
        local_files[path] = (mode, osha)

    changed = {p: v for p, v in local_files.items() if rfiles.get(p) != v}
    deleted = [p for p in rfiles if p not in local_files]
    print(f"changed: {len(changed)}  deleted: {len(deleted)}")

    if not changed and not deleted:
        print("remote already matches local HEAD tree; nothing to push")
        return 0

    entries: list[dict] = []
    n_uploaded = 0
    for i, (path, (mode, lsha)) in enumerate(sorted(changed.items())):
        try:
            existing = gh(f"repos/{args.repo}/git/blobs/{lsha}")
            if existing.get("sha") == lsha:
                entries.append({"path": path, "mode": mode, "type": "blob", "sha": lsha})
                continue
        except RuntimeError:
            pass
        data: bytes = git("show", f"HEAD:{path}", binary=True, check=False)  # type: ignore[assignment]
        if not data and path not in local_files:
            continue
        b64 = base64.b64encode(data).decode()
        out = gh("-X", "POST", f"repos/{args.repo}/git/blobs", stdin=json.dumps({"content": b64, "encoding": "base64"}))
        if out.get("sha") != lsha:
            raise RuntimeError(f"blob sha mismatch for {path}: remote {out.get('sha')} vs local {lsha}")
        n_uploaded += 1
        entries.append({"path": path, "mode": mode, "type": "blob", "sha": lsha})
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(changed)} files processed")
    for path in deleted:
        entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
    print(f"blobs uploaded: {n_uploaded} (reused {len(changed) - n_uploaded})")

    if args.dry_run:
        print("dry-run: would push", entries[:5], "...")
        return 0

    tree = gh("-X", "POST", f"repos/{args.repo}/git/trees",
              stdin=json.dumps({"base_tree": rtree_sha, "tree": entries}))["sha"]
    an = str(git("log", "-1", "--format=%an"))
    ae = str(git("log", "-1", "--format=%ae"))
    ad = str(git("log", "-1", "--format=%aI"))
    msg = args.message or subject + ("\n\n" + body if body else "")
    msg += f"\n\n[pushed via Git Data API; local commit {head}]"
    payload = {"message": msg, "tree": tree, "parents": [remote_sha],
               "author": {"name": an, "email": ae, "date": ad}}
    new_sha = gh("-X", "POST", f"repos/{args.repo}/git/commits", stdin=json.dumps(payload))["sha"]
    gh("-X", "PATCH", f"repos/{args.repo}/git/refs/heads/{branch}", stdin=json.dumps({"sha": new_sha, "force": False}))
    print(f"pushed {new_sha} -> refs/heads/{branch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
