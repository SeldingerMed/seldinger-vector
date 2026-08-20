"""Kernel-owned evidence references and locator resolution.

A gate outcome must be traceable to *evidence* the kernel can resolve and
hash itself — never merely a status string a verifier self-reports. This
module owns the canonical :class:`EvidenceReference` schema (re-exported by
:mod:`or_audit.eval.trace`) and the resolver that turns a locator into a
stable ``uri`` plus a kernel-computed digest.

Locator grammar:

* A bare dotted path or JSON pointer — resolved against the scoring
  ``context`` dict. A leading ``context:`` prefix is accepted and stripped.
* ``task://<path>`` — a package-relative file inside the task package; the
  stored ``uri`` is the stable ``task://…`` reference, never a host path.
  A ``#/json/pointer`` fragment selects a scalar sub-value (used to bind
  named safety thresholds to versioned package artifacts).

Digests are RFC-8785 canonical SHA-256 via :mod:`or_audit.audit.canonical`
for structured values, and raw-byte SHA-256 for ``task://`` files, so a
given evidence value always hashes identically across processes and replay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from or_audit.audit.canonical import digest as canonical_digest
from or_audit.errors import TaskContractError

_MISSING = object()


class EvidenceReference(BaseModel):
    """A resolved, kernel-hashed piece of evidence behind an outcome.

    One canonical schema serves both procedural traces (bound by ``id``/
    ``uri``/``digest``/``media_type``) and gate outcomes (``signal`` names
    the DSL binding, ``uri`` is the stable locator).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    #: Stable locator: dotted path, JSON pointer, or ``task://…`` URI. Never
    #: an absolute host path, so traces and scorecards stay relocatable.
    uri: str
    digest: str = ""
    media_type: str = ""
    #: Name this evidence is bound to within a DSL gate expression ("" for
    #: trace evidence).
    signal: str = ""
    #: Optional display hint; never an absolute host path.
    location: str = ""


def _resolve_dotted(context: Any, path: str) -> Any:
    """Resolve a dotted (or JSON-pointer) path against ``context``."""
    if path.startswith("/"):
        tokens = [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]
    else:
        tokens = path.split(".")
    current = context
    for token in tokens:
        if not token:
            continue
        if isinstance(current, dict):
            if token not in current:
                return _MISSING
            current = current[token]
        elif isinstance(current, (list, tuple)):
            if not token.isdigit() or int(token) < 0 or int(token) >= len(current):
                return _MISSING
            current = current[int(token)]
        else:
            return _MISSING
    return current


def _resolve_pointer_at(value: Any, pointer: str) -> Any:
    """Resolve a ``/a/b/0`` JSON pointer against ``value``."""
    current = value
    for part_raw in pointer[1:].split("/"):
        if not part_raw:
            continue
        part = part_raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
        elif isinstance(current, (list, tuple)):
            if not part.isdigit() or int(part) < 0 or int(part) >= len(current):
                return _MISSING
            current = current[int(part)]
        else:
            return _MISSING
    return current


def normalize_locator(locator: str) -> str:
    """Strip an optional ``context:`` prefix; returns the bare locator."""
    return locator[len("context:") :] if locator.startswith("context:") else locator


def resolve_evidence(
    locator: str,
    *,
    context: dict[str, Any],
    task_root: Path | None = None,
) -> tuple[Any, str, str]:
    """Resolve ``locator`` to ``(value, uri, digest)``.

    ``uri`` is the stable locator (dotted path / JSON pointer / ``task://``
    URI), never a host path. Raises :class:`TaskContractError` for
    out-of-bound, unreadable, or dangling evidence so a fabricated
    reference is a hard error, not a decorative string.
    """
    locator = normalize_locator(locator)
    if locator.startswith("task://"):
        if task_root is None:
            raise TaskContractError(f"evidence locator {locator!r} requires a task package")
        uri, _, fragment = locator.partition("#")
        rel = Path(uri[len("task://") :])
        path = (task_root / rel).resolve()
        try:
            path.relative_to(task_root.resolve())
        except ValueError as exc:
            raise TaskContractError(
                f"evidence locator {locator!r} escapes the task package"
            ) from exc
        if not path.is_file():
            raise TaskContractError(f"evidence file {locator!r} does not exist")
        data = path.read_bytes()
        if fragment:
            try:
                payload = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise TaskContractError(
                    f"evidence file {locator!r} is not JSON for pointer {fragment}"
                ) from exc
            value = _resolve_pointer_at(payload, fragment)
            if value is _MISSING:
                raise TaskContractError(f"evidence pointer {fragment} not found in {locator!r}")
            return value, locator, hashlib.sha256(data).hexdigest()
        return None, locator, hashlib.sha256(data).hexdigest()
    value = _resolve_dotted(context, locator)
    if value is _MISSING:
        # Missing per-trial context evidence is legal abstention/unassessable,
        # not a hard error. ``evaluate_gate`` maps this to NOT_ASSESSABLE.
        return _MISSING, locator, ""
    return value, locator, canonical_digest(value)


def resolve_binding(
    locator: str,
    *,
    context: dict[str, Any],
    task_root: Path | None = None,
    absent_default: Any = _MISSING,
) -> tuple[Any, str, str]:
    """Resolve a gate binding, applying an absent-default where declared.

    A gate may declare an absent-default for an oracle/env boolean field whose
    environment contract defines "absent == a known value" (e.g. a divergence
    flag that is authoritatively ``False`` when unset). When the context path is
    absent **and** a default is supplied, the default is the resolved evidence
    value and is digested like any other binding, so the verdict is still
    kernel-backed rather than verifier-reported. When no default is declared,
    absence propagates as abstention (``_MISSING``).
    """
    value, uri, digest = resolve_evidence(locator, context=context, task_root=task_root)
    if value is _MISSING and absent_default is not _MISSING:
        return absent_default, uri, canonical_digest(absent_default)
    return value, uri, digest
