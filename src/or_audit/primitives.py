"""Shared scalar types.

Single home for constrained strings used by more than one layer, so a pattern
cannot drift between the module that writes a value and the module that
validates it.
"""

from __future__ import annotations

import re
from typing import Annotated, Final

from pydantic import AfterValidator, StringConstraints

#: Lowercase hex SHA-256 digest.
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

#: Every entity prefix the system mints. Closed on purpose: a well-formed but
#: unknown prefix such as ``mrn_`` should fail validation rather than sail
#: through the one structural check the audit trail performs.
ENTITY_PREFIXES: Final = (
    "ins",
    "sur",
    "sys",
    "prc",
    "epi",
    "med",
    "rat",
    "ann",
    "scr",
    "dec",
    "con",
)

#: Body of an identifier: 26 Crockford base32 characters (I, L, O, U excluded).
ID_BODY_PATTERN: Final = r"[0-9A-HJKMNP-TV-Z]{26}"

#: Any identifier this system mints, without pinning which entity. Used where
#: a field legitimately references more than one entity type.
ENTITY_ID_PATTERN: Final = rf"^(?:{'|'.join(ENTITY_PREFIXES)})_{ID_BODY_PATTERN}$"

AnyEntityId = Annotated[str, StringConstraints(pattern=ENTITY_ID_PATTERN)]

_DIGITS_ONLY = re.compile(r"^[0-9][0-9._-]*$")


def _reject_bare_identifier_numbers(value: str) -> str:
    """Reject refs that are only digits and separators.

    Catches the shapes most likely to be a medical record or national
    insurance number pasted into an operator field. Narrow control, not a
    general PHI filter -- see :data:`PrincipalRef`.
    """
    if _DIGITS_ONLY.match(value):
        msg = (
            f"principal ref {value!r} looks like a bare identifier number; "
            f"use a pseudonymous handle that is not purely numeric"
        )
        raise ValueError(msg)
    return value


#: A machine-safe principal reference: lowercase slug, no spaces.
#:
#: What this guarantees, stated plainly because PLAN.md section 9 means a
#: privacy office will test the claim rather than read it: the pattern rejects
#: whitespace, capitals, and most natural renderings of a personal name, and
#: the validator additionally rejects bare identifier numbers such as
#: ``12345678`` or ``123-45-6789``.
#:
#: What it does NOT guarantee: a determined caller can still write
#: ``john.smith``. No regex separates that from a legitimate service name.
#: Keeping personal identifiers out of principal refs is a caller obligation,
#: enforced by policy and by the pseudonymous handles issued upstream. This
#: type narrows the blast radius; it does not close it.
PrincipalRef = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$"),
    AfterValidator(_reject_bare_identifier_numbers),
]
