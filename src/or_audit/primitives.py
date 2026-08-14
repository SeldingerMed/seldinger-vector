"""Shared scalar types.

Single home for constrained strings used by more than one layer, so a pattern
cannot drift between the module that writes a value and the module that
validates it.
"""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import StringConstraints

#: Lowercase hex SHA-256 digest.
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

#: Regex source for an opaque prefixed entity identifier, e.g. ``epi_01H...``.
#: Kept here rather than in ``domain.ids`` because the audit layer validates
#: against it too, and the audit trail must not accept a free-text subject.
ENTITY_ID_PATTERN: Final = r"^[a-z]{3}_[0-9A-HJKMNP-TV-Z]{26}$"

#: Any entity identifier, without pinning which entity. Used where a field
#: legitimately references more than one entity type.
AnyEntityId = Annotated[str, StringConstraints(pattern=ENTITY_ID_PATTERN)]

#: A machine-safe principal reference: lowercase slug, no spaces, no
#: punctuation beyond ``.``, ``_`` and ``-``. Deliberately cannot hold a
#: human name, because the audit trail is append-only and exportable and
#: PLAN.md section 8 keeps personal identifiers out of it.
PrincipalRef = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")]
