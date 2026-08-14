"""Deterministic, cross-language serialization for hashing.

Audit-chain entries and attestation artifacts are hashed, so their byte
representation must be stable across processes, interpreter versions, **and
implementations**. That last requirement is not cosmetic: PLAN.md sections 7.3
and 9 contemplate the chain being re-verified by someone other than us, under
challenge. A canonical form only CPython can reproduce is a materially weaker
claim than one a third party can check in any language.

The output therefore follows RFC 8785 (JSON Canonicalization Scheme):

* Object keys sorted by UTF-16 code unit, which for our key space (ASCII
  identifiers) coincides with code-point order.
* No insignificant whitespace.
* UTF-8, no ASCII escaping, so equal strings hash equally regardless of
  whether they happen to be ASCII.
* Numbers serialized per ECMAScript ``Number::toString``. This is the part
  ``json.dumps`` gets wrong for canonical purposes: Python renders ``1.0`` as
  ``"1.0"`` and ``1e16`` as ``"1e+16"``, where JCS requires ``"1"`` and
  ``"10000000000000000"``.

Rejected rather than silently mangled:

* Non-finite floats. ``NaN`` is not equal to itself, so a digest containing it
  is meaningless, and ``Infinity`` is not valid JSON.
* Integers outside the exact IEEE-754 double range. JSON numbers are doubles;
  a larger integer cannot survive a round trip through a conforming parser, so
  accepting it would produce a digest another implementation cannot reproduce.
* Naive datetimes. An instant without an offset is ambiguous across writers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from math import isfinite
from typing import Any, Final

#: Largest integer exactly representable as an IEEE-754 double.
MAX_SAFE_INTEGER: Final = 2**53 - 1

#: ECMAScript switches to exponential notation outside this decimal-point
#: range (ECMA-262, Number::toString).
_EXP_LOWER: Final = -6
_EXP_UPPER: Final = 21


def _format_number(value: int | float) -> str:
    """Serialize a number per ECMAScript ``Number::toString``.

    Args:
        value: A finite ``int`` within the safe range, or a finite ``float``.

    Returns:
        The canonical textual form.

    Raises:
        ValueError: If the value is non-finite or an out-of-range integer.
    """
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            msg = (
                f"integer {value} exceeds the exact IEEE-754 double range; "
                f"it cannot round-trip through a conforming JSON parser, so "
                f"no cross-implementation canonical form exists"
            )
            raise ValueError(msg)
        return str(value)

    if not isfinite(value):
        msg = f"non-finite float {value!r} cannot be canonicalized"
        raise ValueError(msg)
    if value == 0:
        # Collapses -0.0 to "0", matching ECMAScript.
        return "0"

    negative = value < 0
    # ``repr`` gives the shortest round-tripping decimal; Decimal then lets us
    # reposition the point exactly, without reintroducing binary error.
    sign, digit_tuple, exponent = Decimal(repr(abs(value))).as_tuple()
    del sign
    digits = "".join(str(d) for d in digit_tuple).rstrip("0") or "0"
    significant = len(digits)
    # Value equals 0.<digits> x 10**point, i.e. the decimal point sits after
    # ``point`` digits.
    point = len(digit_tuple) + int(exponent)

    if significant <= point <= _EXP_UPPER:
        text = digits + "0" * (point - significant)
    elif 0 < point <= _EXP_UPPER:
        text = f"{digits[:point]}.{digits[point:]}"
    elif _EXP_LOWER < point <= 0:
        text = "0." + "0" * -point + digits
    else:
        power = point - 1
        mantissa = digits[0] + (f".{digits[1:]}" if significant > 1 else "")
        text = f"{mantissa}e{'+' if power >= 0 else '-'}{abs(power)}"

    return f"-{text}" if negative else text


def _escape(text: str) -> str:
    """Escape a string per RFC 8785 (the JSON minimal-escape set)."""
    out = ['"']
    for char in text:
        code = ord(char)
        if char == '"':
            out.append('\\"')
        elif char == "\\":
            out.append("\\\\")
        elif char == "\b":
            out.append("\\b")
        elif char == "\f":
            out.append("\\f")
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def _render(value: Any) -> str:
    """Render ``value`` as canonical JSON text."""
    if isinstance(value, Enum):
        return _render(value.value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _escape(value)
    if isinstance(value, int | float):
        return _format_number(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            msg = "naive datetime cannot be canonicalized; attach a timezone"
            raise ValueError(msg)
        # Normalize to UTC so equal instants serialize identically regardless
        # of the offset they were expressed in.
        return _escape(value.astimezone(UTC).isoformat().replace("+00:00", "Z"))
    if isinstance(value, Mapping):
        items = []
        for key in sorted(value):
            if not isinstance(key, str):
                msg = f"object keys must be strings for canonical form, got {type(key).__name__}"
                raise TypeError(msg)
            items.append(f"{_escape(key)}:{_render(value[key])}")
        return "{" + ",".join(items) + "}"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return "[" + ",".join(_render(item) for item in value) + "]"
    msg = f"type {type(value).__name__} has no canonical JSON form"
    raise TypeError(msg)


def canonical_json(value: Any) -> str:
    """Render ``value`` as canonical JSON text."""
    return _render(value)


def canonical_bytes(value: Any) -> bytes:
    """Render ``value`` as canonical JSON encoded UTF-8."""
    return canonical_json(value).encode("utf-8")


def digest(value: Any) -> str:
    """Return the lowercase hex SHA-256 of ``value`` in canonical form."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def digest_text(text: str) -> str:
    """Return the lowercase hex SHA-256 of already-canonical ``text``.

    Used where the canonical bytes are stored verbatim, so the digest is
    derived from exactly what was persisted rather than from a re-rendering.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
