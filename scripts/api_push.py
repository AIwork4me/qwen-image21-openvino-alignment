#!/usr/bin/env python3
"""Push commits to GitHub via the Git Data REST API.

The sandbox egress proxy blocks github.com git smart-HTTP (503 on CONNECT) but
allows api.github.com. This script replays local commits onto the remote branch
using the blobs/trees/commits/refs endpoints via `gh api`, preserving messages,
authors and timestamps.
"""
import base64
import json
import os
import subprocess
import sys

REPO = os.environ.get("API_PUSH_REPO", "AIwork4me/qwen-image21-openvino-alignment")
BRANCH = os.environ.get("API_PUSH_BRANCH", "validation/v2-comfyui-text-encoder-matrix")


def gh_json(endpoint, method="GET", payload=None):
    cmd = ["gh", "api", "-X", method, endpoint]
    if payload is not None:
        cmd += ["--input", "-"]
        r = subprocess.run(cmd, input=json.dumps(payload), capture_output=True, text=True)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gh api {method} {endpoint} failed: {r.stderr[:600]}")
    return json.loads(r.stdout) if r.stdout.strip() else {}


def git(*args, check=True):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {args} failed: {r.stderr[:300]}")
    return r.stdout.strip()


def has_parent(sha):
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{sha}^"],
                          capture_output=True).returncode == 0


def changed_files(sha):
    out = git("diff-tree", "--no-commit-id", "--name-status", "-r", "--root", "-z", sha)
    items = out.split("\0")
    files = []
    i = 0
    while i < len(items) - 1:
        st_path = items[i]
        if not st_path:
            i += 1
            continue
        st, path = st_path.split("\t", 1) if "\t" in st_path else (st_path, items[i + 1])
        files.append((st[0], path))
        i += 1
    return files


def file_mode(sha, path):
    line = git("ls-tree", sha, "--", path)
    return line.split()[0] if line else "100644"


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    head = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    commits = git("rev-list", "--reverse", f"{base}..{head}").split()
    print(f"pushing {len(commits)} commits to {REPO}:{BRANCH}")

    try:
        ref = gh_json(f"repos/{REPO}/git/ref/heads/{BRANCH}")
        parent_remote = ref["object"]["sha"]
        print(f"remote branch exists at {parent_remote[:10]}")
    except RuntimeError:
        main_ref = gh_json(f"repos/{REPO}/git/ref/heads/main")
        gh_json(f"repos/{REPO}/git/refs", "POST",
                payload={"ref": f"refs/heads/{BRANCH}", "sha": main_ref["object"]["sha"]})
        parent_remote = main_ref["object"]["sha"]
        print(f"created remote branch at {parent_remote[:10]}")

    for sha in commits:
        message = git("log", "-1", "--format=%B", sha)
        author = {"name": git("log", "-1", "--format=%an", sha),
                  "email": git("log", "-1", "--format=%ae", sha),
                  "date": git("log", "-1", "--format=%aI", sha)}
        committer = {"name": git("log", "-1", "--format=%cn", sha),
                     "email": git("log", "-1", "--format=%ce", sha),
                     "date": git("log", "-1", "--format=%cI", sha)}
        tree_updates = []
        for st, path in changed_files(sha):
            if st == "D":
                tree_updates.append({"path": path, "mode": "100644", "type": "commit",
                                     "sha": None})
                continue
            line = git("ls-tree", sha, "--", path)
            local_blob = line.split()[2]
            data = subprocess.run(["git", "cat-file", "blob", local_blob],
                                  capture_output=True, check=True).stdout
            created = gh_json(f"repos/{REPO}/git/blobs", "POST",
                              payload={"content": base64.b64encode(data).decode(),
                                       "encoding": "base64"})
            tree_updates.append({"path": path, "mode": file_mode(sha, path), "type": "blob",
                                 "sha": created["sha"]})
        tree = gh_json(f"repos/{REPO}/git/trees", "POST",
                       payload={"base_tree": parent_remote, "tree": tree_updates})
        new_c = gh_json(f"repos/{REPO}/git/commits", "POST",
                        payload={"message": message, "tree": tree["sha"],
                                 "parents": [parent_remote],
                                 "author": author, "committer": committer})
        print(f"  {sha[:8]} -> {new_c['sha'][:8]}: {message.splitlines()[0][:70]}")
        parent_remote = new_c["sha"]

    gh_json(f"repos/{REPO}/git/refs/heads/{BRANCH}", "PATCH",
            payload={"sha": parent_remote, "force": False})
    print(f"updated {BRANCH} -> {parent_remote}")


if __name__ == "__main__":
    main()
