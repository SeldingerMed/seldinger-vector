"""Typed, prefixed, lexicographically sortable identifiers.

Identifiers are ULID-style: a 48-bit millisecond timestamp followed by 80
bits of randomness, rendered in Crockford base32. This gives us three
properties the platform needs:

* **Sortable.** Audit review reads better in creation order.
* **Prefixed.** ``sur_`` vs ``epi_`` makes cross-entity mix-ups a validation
  error rather than a silent lookup miss.
* **Opaque.** No embedded institution, name, or MRN. Identifiers are safe to
  appear in logs and in exported artifacts.

No external dependency: the encoding is ~20 lines and pinning a ULID library
for it would add more surface than it removes.
"""

from __future__ import annotations

import os
import time
from typing import Annotated, Final

from pydantic import StringConstraints

# Crockford base32, excluding I, L, O and U to avoid transcription ambiguity.
_ALPHABET: Final = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ENCODED_LEN: Final = 26
_TIMESTAMP_BITS: Final = 48
_RANDOM_BITS: Final = 80

# Character class matching ``_ALPHABET`` for identifier validation.
_ID_CHARS: Final = "[0-9A-HJKMNP-TV-Z]"


def _constraint(prefix: str) -> StringConstraints:
    """Build the pydantic constraint pinning a string to one identifier prefix."""
    return StringConstraints(pattern=rf"^{prefix}_{_ID_CHARS}{{{_ENCODED_LEN}}}$")


def _encode(value: int) -> str:
    """Render a 128-bit integer as 26 Crockford base32 characters."""
    chars = [""] * _ENCODED_LEN
    for index in range(_ENCODED_LEN - 1, -1, -1):
        chars[index] = _ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(chars)


def mint(prefix: str) -> str:
    """Mint a new prefixed identifier.

    Args:
        prefix: Short entity tag, e.g. ``"epi"``. Lowercase ASCII letters only.

    Returns:
        An identifier of the form ``<prefix>_<26 base32 chars>``.

    Raises:
        ValueError: If ``prefix`` is not lowercase ASCII letters.
    """
    if not prefix.isascii() or not prefix.isalpha() or not prefix.islower():
        msg = f"identifier prefix must be lowercase ASCII letters, got {prefix!r}"
        raise ValueError(msg)
    timestamp_ms = int(time.time() * 1000) & ((1 << _TIMESTAMP_BITS) - 1)
    randomness = int.from_bytes(os.urandom(_RANDOM_BITS // 8), "big")
    return f"{prefix}_{_encode((timestamp_ms << _RANDOM_BITS) | randomness)}"


# Entity identifier types. Each is a plain ``str`` at runtime, so it
# serializes transparently, but pydantic rejects a mismatched prefix at the
# model boundary -- which is where cross-entity confusion actually happens.
InstitutionId = Annotated[str, _constraint("ins")]
SurgeonId = Annotated[str, _constraint("sur")]
RoboticSystemId = Annotated[str, _constraint("sys")]
ProcedureId = Annotated[str, _constraint("prc")]
EpisodeId = Annotated[str, _constraint("epi")]
MediaAssetId = Annotated[str, _constraint("med")]
RaterId = Annotated[str, _constraint("rat")]
AnnotationId = Annotated[str, _constraint("ann")]
ScoreId = Annotated[str, _constraint("scr")]
DecisionId = Annotated[str, _constraint("dec")]
ContestationId = Annotated[str, _constraint("con")]


def new_institution_id() -> InstitutionId:
    """Mint an institution identifier."""
    return mint("ins")


def new_surgeon_id() -> SurgeonId:
    """Mint a surgeon identifier."""
    return mint("sur")


def new_system_id() -> RoboticSystemId:
    """Mint a robotic-system identifier."""
    return mint("sys")


def new_procedure_id() -> ProcedureId:
    """Mint a procedure identifier."""
    return mint("prc")


def new_episode_id() -> EpisodeId:
    """Mint an episode identifier."""
    return mint("epi")


def new_media_asset_id() -> MediaAssetId:
    """Mint a media-asset identifier."""
    return mint("med")


def new_rater_id() -> RaterId:
    """Mint a rater identifier."""
    return mint("rat")


def new_annotation_id() -> AnnotationId:
    """Mint an annotation identifier."""
    return mint("ann")


def new_score_id() -> ScoreId:
    """Mint a score identifier."""
    return mint("scr")


def new_decision_id() -> DecisionId:
    """Mint a decision identifier."""
    return mint("dec")


def new_contestation_id() -> ContestationId:
    """Mint a contestation identifier."""
    return mint("con")
