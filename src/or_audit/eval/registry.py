"""Versioned public registry for taskset and agent packages."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from or_audit.errors import TaskContractError
from or_audit.eval.integrity import tree_digest

DEFAULT_REGISTRY = (
    "https://raw.githubusercontent.com/SeldingerMed/seldinger-tasks/main/registry.json"
)
RegistryId = Annotated[
    str,
    StringConstraints(
        min_length=3,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*$",
    ),
]
RegistryVersion = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]


class RegistryEntry(BaseModel):
    """One immutable taskset or agent package."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["taskset", "agent"]
    id: RegistryId
    version: RegistryVersion
    repository: str
    ref: str
    path: str
    digest: str

    @model_validator(mode="before")
    @classmethod
    def _normalize_dataset_kind(cls, raw: Any) -> Any:
        if isinstance(raw, dict) and raw.get("kind") == "dataset":
            return {**raw, "kind": "taskset"}
        return raw

    @property
    def reference(self) -> str:
        return f"{self.id}@{self.version}"


class RegistryIndex(BaseModel):
    """Public registry index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: Literal["1"]
    tasksets: tuple[RegistryEntry, ...] = ()
    agents: tuple[RegistryEntry, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _normalize_datasets(cls, raw: Any) -> Any:
        if isinstance(raw, dict) and "tasksets" not in raw and "datasets" in raw:
            data = dict(raw)
            data["tasksets"] = data.pop("datasets")
            return data
        return raw

    @property
    def datasets(self) -> tuple[RegistryEntry, ...]:
        return self.tasksets


def load_registry(source: str = DEFAULT_REGISTRY) -> RegistryIndex:
    """Load a local path, ``file://`` path, or HTTPS registry index."""
    try:
        if source.startswith(("https://", "http://")):
            with urllib.request.urlopen(source, timeout=30) as response:
                payload = json.load(response)
        else:
            raw = source.removeprefix("file://")
            payload = json.loads(Path(raw).read_text(encoding="utf-8"))
        return RegistryIndex.model_validate(payload)
    except TaskContractError:
        raise
    except Exception as exc:
        raise TaskContractError(f"failed to load registry {source}: {exc}") from exc


def resolve_entry(
    index: RegistryIndex,
    *,
    kind: Literal["taskset", "dataset", "agent"],
    ref: str,
) -> RegistryEntry:
    """Resolve exact ``org/name@version`` identity; floating versions are refused."""
    canonical_kind = "taskset" if kind == "dataset" else kind
    entries = index.tasksets if canonical_kind == "taskset" else index.agents
    matches = [entry for entry in entries if entry.reference == ref]
    if len(matches) != 1:
        known = ", ".join(sorted(entry.reference for entry in entries)) or "(none)"
        raise TaskContractError(f"unknown {kind} {ref!r}; known: {known}")
    entry = matches[0]
    if entry.kind != canonical_kind:
        raise TaskContractError(
            f"registry entry {ref} has kind={entry.kind}, expected {canonical_kind}"
        )
    return entry


def _checkout(entry: RegistryEntry, cache_root: Path) -> Path:
    repository = entry.repository.removeprefix("file://")
    local = Path(repository).expanduser()
    if local.exists():
        return local.resolve()
    key = hashlib.sha256(f"{entry.repository}@{entry.ref}".encode()).hexdigest()[:20]
    checkout = cache_root / key
    if not checkout.is_dir():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                entry.repository,
                str(checkout),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", entry.ref],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", entry.ref],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    expected = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", entry.ref],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if actual != expected:
        raise TaskContractError(
            f"registry checkout mismatch for {entry.reference}: expected {expected}, got {actual}"
        )
    return checkout


def materialize_entry(
    entry: RegistryEntry,
    *,
    cache_root: Path | None = None,
) -> Path:
    """Resolve and content-verify one immutable package directory."""
    root = _checkout(entry, cache_root or Path.home() / ".cache" / "or-audit" / "registry")
    package = (root / entry.path).resolve()
    try:
        package.relative_to(root.resolve())
    except ValueError as exc:
        raise TaskContractError(f"registry path escapes repository: {entry.path!r}") from exc
    actual = tree_digest(package)
    if actual != entry.digest:
        raise TaskContractError(
            f"registry digest mismatch for {entry.reference}: "
            f"declared {entry.digest}, actual {actual}"
        )
    return package


def pull_entry(entry: RegistryEntry, out: Path) -> Path:
    """Copy a verified package into a caller-owned directory."""
    source = materialize_entry(entry)
    target = out / entry.id / entry.version
    shutil.copytree(source, target, dirs_exist_ok=True)
    if tree_digest(target) != entry.digest:
        raise TaskContractError(f"pulled package digest changed for {entry.reference}")
    return target
