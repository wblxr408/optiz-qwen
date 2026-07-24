"""Small, dependency-free helpers shared by D-direction experiment scripts."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def inspect_git(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> tuple[bool, str]:
        clean_env = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("GIT_")
        }
        clean_env["GIT_OPTIONAL_LOCKS"] = "0"
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                env=clean_env,
            )
        except (OSError, subprocess.SubprocessError):
            return False, ""
        return True, (completed.stdout or "").strip()

    commit_ok, commit = run("rev-parse", "HEAD")
    status_ok, status = run("status", "--porcelain", "--untracked-files=normal")
    root_ok, reported_root = run("rev-parse", "--show-toplevel")
    try:
        root_matches_requested = (
            root_ok and Path(reported_root).resolve() == repo_root.resolve()
        )
    except OSError:
        root_matches_requested = False
    return {
        "git_available": commit_ok and status_ok and root_ok,
        "git_commit": commit or None,
        "git_dirty": bool(status) if status_ok else None,
        "repository_root": reported_root or None,
        "repository_root_matches_requested": root_matches_requested,
    }
