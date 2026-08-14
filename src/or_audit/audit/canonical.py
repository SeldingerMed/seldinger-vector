"""Deterministic serialization for hashing.

Audit-chain entries and attestation artifacts are hashed, so their byte
representation must be stable across processes, interpreter versions, and
dict insertion order. ``json.dumps`` with default settings is none of those
things.

Rules enforced here:

* Object keys sorted lexicographically by code point.
* No insignificant whitespace.
* UTF-8, no ASCII escaping, so equal strings hash equally regardless of
  whether they happen to be ASCII.
* Non-finite floats rejected. ``NaN`` is not equal to itself, which makes a
  hash containing it meaningless, and ``Infinity`` is not valid JSON.
* Only JSON-native types plus ``datetime`` and ``Enum``, both normalized.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from math import isfinite
from typing import Any

_JSON_SCALARS = (str, bool, int, float, type(None))


def _normalize(value: Any) -> Any:
    """Reduce ``value`` to JSON-native types with stable representations."""
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            msg = "naive datetime cannot be canonicalized; attach a timezone"
            raise ValueError(msg)
        # Normalize to UTC so equal instants serialize identically regardless
        # of the offset they were expressed in.
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, bool | type(None) | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            msg = f"non-finite float {value!r} cannot be canonicalized"
            raise ValueError(msg)
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"object keys must be strings for canonical form, got {type(key).__name__}"
                raise TypeError(msg)
            out[key] = _normalize(item)
        return out
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_normalize(item) for item in value]
    msg = f"type {type(value).__name__} has no canonical JSON form"
    raise TypeError(msg)


def canonical_json(value: Any) -> str:
    """Render ``value`` as canonical JSON text."""
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    """Render ``value`` as canonical JSON encoded UTF-8."""
    return canonical_json(value).encode("utf-8")


def digest(value: Any) -> str:
    """Return the lowercase hex SHA-256 of ``value`` in canonical form."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
