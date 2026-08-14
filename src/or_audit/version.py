"""Component version identifiers.

Every attestation artifact records the versions that produced it, because
PLAN.md section 7.3 requires an immutable audit trail of score version,
model version, and decision-rule version. A score is not interpretable
without knowing what produced it.
"""

from __future__ import annotations

from typing import Final

#: Version of the OR-Audit package as a whole.
PACKAGE_VERSION: Final = "0.1.0a0"

#: Version of the domain schema. Bump on any breaking change to entity or
#: score shapes; persisted records carry this so old artifacts stay readable.
SCHEMA_VERSION: Final = "1"

#: Version of the audit-chain construction (hash inputs and ordering).
#: Bump invalidates chain verification of older logs, so treat as frozen.
AUDIT_CHAIN_VERSION: Final = "1"
