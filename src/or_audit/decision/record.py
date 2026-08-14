"""Determinations and the contestation machinery around them.

PLAN.md section 7.3 requires contestability as a product requirement rather
than a nicety, and gives the reason plainly: the subject of an adverse score is
a licensed professional with career exposure and a strong incentive to
litigate. A determination that cannot be examined, answered, or appealed is
one that will be attacked as a whole.

Five things are therefore built in rather than deferred:

* Right of access to the determination and the evidence behind it.
* A defined appeals path routing to human expert re-review.
* **Rater disagreement surfaced**, not smoothed. Where the panel split, the
  artifact says so.
* Right of response, recorded durably and attached to the record.
* An immutable version trail: rule version, perception backend, policy
  versions, all pinned at the moment of decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from or_audit.audit.canonical import digest
from or_audit.domain.enums import Determination
from or_audit.domain.ids import DecisionId, EpisodeId, SurgeonId
from or_audit.errors import DomainInvariantError
from or_audit.primitives import PrincipalRef
from or_audit.version import SCHEMA_VERSION

Text = Annotated[str, StringConstraints(min_length=1, max_length=4000)]


class RaterDisagreement(BaseModel):
    """A point where the expert panel did not agree.

    Surfaced rather than averaged away. A determination resting on a 2-1 split
    is a different artifact from one resting on unanimity, and the surgeon --
    and any later reviewer -- is entitled to know which they are looking at.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    #: One entry per rater, in no particular order.
    positions: tuple[str, ...]

    @model_validator(mode="after")
    def _actually_disagreed(self) -> Self:
        if len(self.positions) < 2:
            msg = "a disagreement needs at least two positions"
            raise DomainInvariantError(msg)
        if len(set(self.positions)) < 2:
            msg = (
                f"positions on {self.subject!r} are unanimous; record a "
                f"disagreement only where the panel actually split"
            )
            raise DomainInvariantError(msg)
        return self


class ContestationState(StrEnum):
    """Where a challenge has got to."""

    FILED = "filed"
    UNDER_REVIEW = "under_review"
    #: Re-review changed the determination.
    UPHELD_FOR_SUBJECT = "upheld_for_subject"
    #: Re-review left the determination standing.
    ORIGINAL_STANDS = "original_stands"
    WITHDRAWN = "withdrawn"


class SubjectResponse(BaseModel):
    """The surgeon's own statement, attached durably to the record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    submitted_at: datetime
    submitted_by: PrincipalRef
    statement: Text

    @model_validator(mode="after")
    def _aware(self) -> Self:
        if self.submitted_at.tzinfo is None:
            msg = "subject response submitted_at must be timezone-aware"
            raise DomainInvariantError(msg)
        return self


class Contestation(BaseModel):
    """A challenge to a determination."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: ContestationState
    filed_at: datetime
    filed_by: PrincipalRef
    grounds: Text
    #: Populated once re-review concludes.
    resolved_at: datetime | None = None
    resolution_note: str | None = None
    #: The determination re-review arrived at, when it differs.
    revised_determination: Determination | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.filed_at.tzinfo is None:
            msg = "contestation filed_at must be timezone-aware"
            raise DomainInvariantError(msg)
        terminal = {
            ContestationState.UPHELD_FOR_SUBJECT,
            ContestationState.ORIGINAL_STANDS,
            ContestationState.WITHDRAWN,
        }
        if self.state in terminal:
            if self.resolved_at is None:
                msg = f"a {self.state.value} contestation must record when it resolved"
                raise DomainInvariantError(msg)
            if self.resolved_at.tzinfo is None:
                msg = "contestation resolved_at must be timezone-aware"
                raise DomainInvariantError(msg)
            if self.resolved_at < self.filed_at:
                msg = "a contestation cannot resolve before it was filed"
                raise DomainInvariantError(msg)
        elif self.resolved_at is not None:
            msg = f"a {self.state.value} contestation is not resolved but names a resolution time"
            raise DomainInvariantError(msg)

        if self.state is ContestationState.UPHELD_FOR_SUBJECT and (
            self.revised_determination is None
        ):
            msg = (
                "a contestation upheld for the subject must name the revised "
                "determination; 'upheld' without a change is not a resolution"
            )
            raise DomainInvariantError(msg)
        if (
            self.revised_determination is not None
            and self.state is not ContestationState.UPHELD_FOR_SUBJECT
        ):
            msg = (
                f"a {self.state.value} contestation names a revised determination; "
                f"only an upheld challenge changes the outcome"
            )
            raise DomainInvariantError(msg)
        return self

    @property
    def is_open(self) -> bool:
        """Whether the challenge is still live."""
        return self.state in {ContestationState.FILED, ContestationState.UNDER_REVIEW}


class DecisionRecord(BaseModel):
    """A determination, everything that produced it, and everything said since.

    The versions are pinned rather than referenced: a record must remain
    interpretable after the rule, the perception backend and the policies have
    all moved on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: DecisionId
    schema_version: str = SCHEMA_VERSION
    episode_id: EpisodeId
    surgeon_id: SurgeonId

    determination: Determination
    reason: Text
    decided_at: datetime
    decided_by: PrincipalRef

    #: Pinned provenance. Section 7.3 requires score, model and rule versions
    #: on the record itself.
    rule_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    perception_identity: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    gate_policy_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]

    #: Where the panel split. Empty means unanimous, not unexamined.
    disagreements: tuple[RaterDisagreement, ...] = ()
    #: Challenges, in filing order.
    contestations: tuple[Contestation, ...] = ()
    #: The surgeon's statements, in submission order.
    responses: tuple[SubjectResponse, ...] = ()

    @model_validator(mode="after")
    def _aware_and_ordered(self) -> Self:
        if self.decided_at.tzinfo is None:
            msg = "decision decided_at must be timezone-aware"
            raise DomainInvariantError(msg)
        if any(c.filed_at < self.decided_at for c in self.contestations):
            msg = "a contestation cannot be filed before the decision it challenges"
            raise DomainInvariantError(msg)
        if any(r.submitted_at < self.decided_at for r in self.responses):
            msg = "a response cannot predate the decision it answers"
            raise DomainInvariantError(msg)
        if sum(1 for c in self.contestations if c.is_open) > 1:
            msg = (
                "only one contestation may be open at a time; concurrent "
                "challenges to the same determination cannot be resolved coherently"
            )
            raise DomainInvariantError(msg)
        return self

    @property
    def digest(self) -> str:
        """Content digest, for binding into the audit trail."""
        return digest(self.model_dump(mode="python"))

    @property
    def is_adverse(self) -> bool:
        """Whether the effective determination is against the surgeon.

        Uses the effective determination, so a successful challenge stops the
        record reading as adverse.
        """
        return self.effective_determination is Determination.DOES_NOT_MEET

    @property
    def effective_determination(self) -> Determination:
        """The determination that currently stands.

        A successful challenge supersedes the original. The original is not
        erased -- it stays in ``determination`` and the contestation explains
        the change -- because rewriting history is exactly what the audit trail
        exists to prevent.
        """
        for contestation in reversed(self.contestations):
            if (
                contestation.state is ContestationState.UPHELD_FOR_SUBJECT
                and contestation.revised_determination is not None
            ):
                return contestation.revised_determination
        return self.determination

    @property
    def was_revised(self) -> bool:
        """Whether a challenge changed the outcome."""
        return self.effective_determination is not self.determination

    @property
    def has_open_contestation(self) -> bool:
        """Whether a challenge is currently live."""
        return any(c.is_open for c in self.contestations)

    @property
    def is_unanimous(self) -> bool:
        """Whether the panel agreed at every recorded point."""
        return not self.disagreements

    def with_contestation(self, contestation: Contestation) -> DecisionRecord:
        """Return a new record with ``contestation`` appended.

        Records are immutable; challenging one produces a new record rather
        than mutating the old, so the sequence of states stays inspectable.

        Rebuilt through the constructor rather than ``model_copy`` so the class
        validators actually run. ``model_copy`` skips them, which let this
        method append a second open challenge or one filed before the decision
        -- records the constructor rejects. A guard the write path can walk
        around is worse than no guard, because it reads as protection.

        Raises:
            DomainInvariantError: If the result would be incoherent.
        """
        return self._rebuilt(contestations=(*self.contestations, contestation))

    def with_response(self, response: SubjectResponse) -> DecisionRecord:
        """Return a new record with the subject's statement appended.

        Validated for the same reason as :meth:`with_contestation`.

        Raises:
            DomainInvariantError: If the response predates the decision.
        """
        return self._rebuilt(responses=(*self.responses, response))

    def _rebuilt(self, **changes: object) -> DecisionRecord:
        """Reconstruct through the constructor so validators run."""
        return DecisionRecord(**{**self.model_dump(mode="python"), **changes})

    def subject_disclosure(self) -> dict[str, object]:
        """Everything the subject is entitled to see.

        Section 7.3's right of access. Deliberately complete: the
        determination, the reason, the versions that produced it, where the
        panel split, and their own responses. Withholding the disagreements
        would make the artifact look more certain than it is.
        """
        return {
            "decision_id": self.id,
            "determination": self.effective_determination.value,
            "original_determination": self.determination.value,
            "was_revised": self.was_revised,
            "reason": self.reason,
            "decided_at": self.decided_at,
            "rule_version": self.rule_version,
            "perception": self.perception_identity,
            "gate_policy_version": self.gate_policy_version,
            "panel_disagreements": [
                {"subject": d.subject, "positions": list(d.positions)} for d in self.disagreements
            ],
            "contestations": [
                {"state": c.state.value, "filed_at": c.filed_at, "grounds": c.grounds}
                for c in self.contestations
            ],
            "responses": [
                {"submitted_at": r.submitted_at, "statement": r.statement} for r in self.responses
            ],
            "appeal_available": not self.has_open_contestation,
        }


def open_contestations(records: Sequence[DecisionRecord]) -> tuple[DecisionRecord, ...]:
    """Records with a live challenge, for a review queue."""
    return tuple(r for r in records if r.has_open_contestation)
