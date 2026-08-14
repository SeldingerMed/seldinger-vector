"""Error taxonomy for OR-Audit.

Errors are split by *who* is responsible, because the response differs:
a `DeidentificationBoundaryError` is a compliance incident, while a
`ScoreContractError` is a programming defect.
"""

from __future__ import annotations


class OrAuditError(Exception):
    """Base class for every error raised by OR-Audit."""


class DomainInvariantError(OrAuditError):
    """A domain object was constructed in a state the plan forbids."""


class DeidentificationBoundaryError(OrAuditError):
    """Attempted use of media that has not cleared de-identification.

    Per PLAN.md section 8, no episode media may be read by perception,
    scoring, export, or reporting until it carries a de-identification
    attestation. Raising this is a compliance-relevant event and callers
    must not suppress it.
    """


class AuditChainError(OrAuditError):
    """The append-only audit chain failed verification."""


class ScoreContractError(OrAuditError):
    """A score vector was combined in a way the plan prohibits.

    Per PLAN.md section 7.1, hard safety gates never average into soft
    scores and the vector is never implicitly collapsed to a scalar.
    """
