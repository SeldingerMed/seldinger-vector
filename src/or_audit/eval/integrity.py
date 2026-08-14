"""Content identity for task, agent, and registry packages."""

from __future__ import annotations

import hashlib
from pathlib import Path

from or_audit.audit.canonical import digest
from or_audit.errors import TaskContractError

_IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".git"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


def package_file(root: Path, relative: str, *, label: str) -> Path:
    """Resolve a package-relative file without allowing traversal."""
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise TaskContractError(f"{label} escapes package root: {relative!r}") from exc
    if not candidate.is_file():
        raise TaskContractError(f"missing {label}: {candidate}")
    return candidate


def file_sha256(path: Path) -> str:
    """Hash file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root: Path) -> str:
    """Hash every durable file by relative path and byte digest."""
    if not root.is_dir():
        raise TaskContractError(f"package root is not a directory: {root}")
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix in _IGNORED_SUFFIXES:
            continue
        rows.append({"path": relative.as_posix(), "sha256": file_sha256(path)})
    if not rows:
        raise TaskContractError(f"package has no durable files: {root}")
    return digest(rows)
