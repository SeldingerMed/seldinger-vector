"""Deterministic hard safety gates.

The bottom layer of the verifier stack in PLAN.md section 7.1. These are rules,
not models: given the same observations they always return the same verdict,
and the reason is always statable in one sentence. That matters because a gate
result can be adverse to a named clinician (section 9), and "the model said so"
is not a defensible answer.

Three properties are enforced rather than documented:

* **Gates never average into soft scores.** :class:`SafetyGateSet` refuses to
  produce a scalar. Section 7.1 says hard gates stay distinct from soft scores,
  and the way to keep a rule is to make breaking it raise.
* **Abstention is a first-class verdict.** Missing or low-confidence evidence
  yields ``NOT_ASSESSABLE``, never ``PASS``. A gate that cannot see cannot
  clear.
* **Applicability is explicit.** The CVS gate applies to cholecystectomy-family
  procedures only. Running it elsewhere reports ``NOT_APPLICABLE`` rather than
  a vacuous pass.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from or_audit.domain.entities import Episode, Procedure
from or_audit.domain.enums import GateStatus, MediaKind
from or_audit.errors import DeidentificationBoundaryError, ScoreContractError
from or_audit.perception.observations import (
    BLEEDING_RANK,
    BleedingSeverity,
    CriticalStructure,
    CvsCriterion,
    ObservationKind,
    PerceptionResult,
    SurgicalPhase,
)

#: Structures whose injury is the never-event this platform screens for.
NEVER_INJURE = frozenset(
    {
        CriticalStructure.COMMON_BILE_DUCT,
        CriticalStructure.COMMON_HEPATIC_DUCT,
        CriticalStructure.RIGHT_HEPATIC_ARTERY,
        CriticalStructure.URETER,
    }
)


class GateId(StrEnum):
    """The gates this module evaluates."""

    CVS = "critical_view_of_safety"
    CRITICAL_STRUCTURE_PROXIMITY = "critical_structure_proximity"
    BLEEDING = "bleeding"


class GateResult(BaseModel):
    """One gate's verdict, with its reason and the evidence behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate: GateId
    status: GateStatus
    #: One sentence a reviewer can act on. Never empty.
    reason: str
    #: Backend identity that produced the observations, for the audit trail.
    perception: str
    #: Structured detail: which criteria held, which events fired.
    evidence: tuple[str, ...] = ()

    @property
    def is_clear(self) -> bool:
        """Whether this gate affirmatively passed.

        ``NOT_ASSESSABLE`` is not clear. The property exists so callers stop
        writing ``status != FAIL``, which silently treats unassessable as fine.
        """
        return self.status is GateStatus.PASS


class SafetyGateSet(BaseModel):
    """All gate results for one episode.

    Deliberately not summable, averageable, or orderable. PLAN.md section 7.1
    requires hard gates to stay distinct from soft scores and the vector never
    to collapse implicitly; a set that quietly supported ``sum()`` would make
    that a convention rather than a rule.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    results: tuple[GateResult, ...]

    def __len__(self) -> int:
        """Number of gates evaluated."""
        return len(self.results)

    def __iter__(self) -> Iterator[GateResult]:  # type: ignore[override]
        """Iterate the gate results.

        Overrides pydantic's default, which yields ``(field_name, value)``
        tuples. That default made ``list(gates)`` return ``[("results", ...)]``
        and ``sum(gates)`` fail with an unhelpful message about tuples, both of
        which invite the caller to conclude the model is misshapen rather than
        that the operation is disallowed.
        """
        return iter(self.results)

    def __float__(self) -> float:
        """Always raises. Gates are not a number."""
        msg = (
            "a safety gate set has no scalar value; hard gates are reported "
            "per gate and never averaged into a score (PLAN.md section 7.1)"
        )
        raise ScoreContractError(msg)

    def __int__(self) -> int:
        """Always raises. Gates are not a number."""
        return int(self.__float__())

    def __bool__(self) -> bool:
        """Always raises.

        ``if gates:`` reads as "did it pass" but would mean "is it non-empty".
        Callers must name the question: :attr:`all_clear`, :attr:`any_failed`,
        or :attr:`unassessable`.
        """
        msg = (
            "a safety gate set has no truth value; ask all_clear, any_failed, "
            "or unassessable so the question is visible at the call site"
        )
        raise ScoreContractError(msg)

    def get(self, gate: GateId) -> GateResult | None:
        """The result for ``gate``, if it was evaluated."""
        return next((r for r in self.results if r.gate is gate), None)

    @property
    def failed(self) -> tuple[GateResult, ...]:
        """Gates that affirmatively failed."""
        return tuple(r for r in self.results if r.status is GateStatus.FAIL)

    @property
    def unassessable(self) -> tuple[GateResult, ...]:
        """Gates that could not be decided on the available evidence."""
        return tuple(r for r in self.results if r.status is GateStatus.NOT_ASSESSABLE)

    @property
    def any_failed(self) -> bool:
        """Whether any gate failed."""
        return bool(self.failed)

    @property
    def all_clear(self) -> bool:
        """Whether every applicable gate affirmatively passed.

        Unassessable gates make this false. That is the whole point: an
        episode nobody could evaluate has not been cleared.
        """
        return all(r.status is GateStatus.PASS for r in self.results)


class GatePolicy(BaseModel):
    """Thresholds governing gate evaluation.

    Versioned and carried onto results, because a verdict is only interpretable
    against the thresholds that produced it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "1"
    #: Confidence at or below which an observation is treated as no evidence.
    #: Set conservatively: a low-confidence "achieved" must not clear a gate.
    min_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6
    #: Distance at or below which an instrument is judged too close.
    proximity_alarm_mm: Annotated[float, Field(gt=0.0)] = 5.0
    #: Bleeding at or above this severity fails the bleeding gate.
    bleeding_fail_at: BleedingSeverity = BleedingSeverity.MAJOR


def evaluate_cvs(result: PerceptionResult, procedure: Procedure, policy: GatePolicy) -> GateResult:
    """Evaluate the Critical View of Safety.

    All three Strasberg criteria must be achieved, with adequate confidence,
    **before** the cystic duct is divided. Achieving them afterwards is not
    achieving them: the point of the view is that it precedes the irreversible
    step.

    Ordering rules:

    * Any criterion explicitly not achieved -> ``FAIL``.
    * Any criterion unknown, or below the confidence floor -> ``NOT_ASSESSABLE``.
    * All achieved but the clipping phase was never observed ->
      ``NOT_ASSESSABLE``, because the timing claim cannot be checked.
    * All achieved after clipping began -> ``FAIL``.
    """
    if not procedure.cvs_applicable:
        return GateResult(
            gate=GateId.CVS,
            status=GateStatus.NOT_APPLICABLE,
            reason=(f"the Critical View of Safety does not apply to {procedure.display_name}"),
            perception=result.identity,
        )

    missing: list[str] = []
    low_confidence: list[str] = []
    not_achieved: list[str] = []
    achieved_at: list[float] = []

    if not result.assessed(ObservationKind.CVS):
        return GateResult(
            gate=GateId.CVS,
            status=GateStatus.NOT_ASSESSABLE,
            reason=(
                f"{result.identity} did not assess the critical view, so nothing "
                f"is known about it either way"
            ),
            perception=result.identity,
        )

    for criterion in CvsCriterion:
        obs = result.cvs_observation(criterion)
        if obs is None or obs.achieved is None:
            missing.append(criterion.value)
            continue
        # The floor is applied BEFORE reading ``achieved``, in both directions.
        # Below it the policy defines the observation as no evidence, and FAIL
        # is the worst verdict available -- resting it on evidence the policy
        # calls absent would be indefensible under section 9 and contradicts
        # this gate's own documented contract.
        if obs.confidence < policy.min_confidence:
            low_confidence.append(f"{criterion.value}({obs.confidence:.2f})")
            continue
        if obs.achieved is False:
            not_achieved.append(criterion.value)
            continue
        # ``at_s`` is guaranteed present when achieved is True.
        achieved_at.append(obs.at_s or 0.0)

    if not_achieved:
        return GateResult(
            gate=GateId.CVS,
            status=GateStatus.FAIL,
            reason=(
                f"the critical view was not achieved: "
                f"{', '.join(sorted(not_achieved))} not satisfied"
            ),
            perception=result.identity,
            evidence=tuple(sorted(not_achieved)),
        )

    if missing or low_confidence:
        detail = []
        if missing:
            detail.append(f"no evidence for {', '.join(sorted(missing))}")
        if low_confidence:
            detail.append(
                f"confidence below {policy.min_confidence} for {', '.join(sorted(low_confidence))}"
            )
        return GateResult(
            gate=GateId.CVS,
            status=GateStatus.NOT_ASSESSABLE,
            reason=f"the critical view could not be assessed: {'; '.join(detail)}",
            perception=result.identity,
            evidence=tuple(sorted(missing) + sorted(low_confidence)),
        )

    clipping = result.phase_span(
        SurgicalPhase.CLIPPING_AND_CUTTING, min_confidence=policy.min_confidence
    )
    if clipping is None:
        return GateResult(
            gate=GateId.CVS,
            status=GateStatus.NOT_ASSESSABLE,
            reason=(
                "all three criteria were satisfied, but the clipping and "
                "cutting phase was never observed, so it cannot be shown that "
                "the view preceded division of the cystic duct"
            ),
            perception=result.identity,
        )

    clipping_start = clipping[0]
    late = max(achieved_at)
    # Strictly before. A view completed at the same instant the irreversible
    # step begins has not been shown to precede it, and passing it would force
    # the reason string to say "satisfied at 200.0s, before clipping began at
    # 200.0s".
    if late >= clipping_start:
        return GateResult(
            gate=GateId.CVS,
            status=GateStatus.FAIL,
            reason=(
                f"the critical view was completed at {late:.3f}s and clipping and "
                f"cutting began at {clipping_start:.3f}s; the view must be complete "
                f"strictly before division of the cystic duct"
            ),
            perception=result.identity,
            evidence=(f"completed_at={late:.1f}s", f"clipping_start={clipping_start:.1f}s"),
        )

    return GateResult(
        gate=GateId.CVS,
        status=GateStatus.PASS,
        reason=(
            f"all three Strasberg criteria were satisfied by {late:.1f}s, before "
            f"clipping began at {clipping_start:.1f}s"
        ),
        perception=result.identity,
        evidence=tuple(c.value for c in CvsCriterion),
    )


def evaluate_proximity(result: PerceptionResult, policy: GatePolicy) -> GateResult:
    """Flag instrument approaches to structures that must not be injured.

    An event whose distance is unmeasured is not treated as safe. It is
    reported as unassessable, because "we could not measure it" and "it was
    far away" are different claims.
    """
    if not result.assessed(ObservationKind.PROXIMITY):
        return GateResult(
            gate=GateId.CRITICAL_STRUCTURE_PROXIMITY,
            status=GateStatus.NOT_ASSESSABLE,
            reason=(
                f"{result.identity} did not assess instrument proximity, so no "
                f"absence of near-misses has been established"
            ),
            perception=result.identity,
        )

    relevant = [e for e in result.proximity_events if e.structure in NEVER_INJURE]
    confident = [e for e in relevant if e.confidence >= policy.min_confidence]
    # Below-floor observations are NOT discarded. Dropping a 1mm approach to
    # the common bile duct because the detector was only 10% sure, then
    # reporting "no instrument approached within the alarm distance", is a
    # false statement on an artifact that can be adverse to a clinician. The
    # gate cannot see clearly, so it does not clear.
    uncertain = [e for e in relevant if e.confidence < policy.min_confidence]

    alarms = [
        e
        for e in confident
        if e.distance_mm is not None and e.distance_mm <= policy.proximity_alarm_mm
    ]
    if alarms:
        return GateResult(
            gate=GateId.CRITICAL_STRUCTURE_PROXIMITY,
            status=GateStatus.FAIL,
            reason=(
                f"{len(alarms)} instrument approach(es) within "
                f"{policy.proximity_alarm_mm}mm of a structure that must not be injured"
            ),
            perception=result.identity,
            evidence=tuple(f"{e.structure.value}@{e.at_s:.1f}s={e.distance_mm}mm" for e in alarms),
        )

    unmeasured = [e for e in confident if e.distance_mm is None]
    if unmeasured or uncertain:
        detail = []
        if unmeasured:
            detail.append(f"{len(unmeasured)} approach(es) without a measurable distance")
        if uncertain:
            detail.append(
                f"{len(uncertain)} approach(es) below the {policy.min_confidence} confidence floor"
            )
        return GateResult(
            gate=GateId.CRITICAL_STRUCTURE_PROXIMITY,
            status=GateStatus.NOT_ASSESSABLE,
            reason=f"instrument proximity could not be assessed: {'; '.join(detail)}",
            perception=result.identity,
            evidence=tuple(f"{e.structure.value}@{e.at_s:.1f}s" for e in unmeasured + uncertain),
        )

    return GateResult(
        gate=GateId.CRITICAL_STRUCTURE_PROXIMITY,
        status=GateStatus.PASS,
        reason="no instrument approached a critical structure within the alarm distance",
        perception=result.identity,
    )


def evaluate_bleeding(result: PerceptionResult, policy: GatePolicy) -> GateResult:
    """Fail on bleeding at or above the configured severity."""
    if not result.assessed(ObservationKind.BLEEDING):
        return GateResult(
            gate=GateId.BLEEDING,
            status=GateStatus.NOT_ASSESSABLE,
            reason=(
                f"{result.identity} did not assess bleeding, so no absence of it "
                f"has been established"
            ),
            perception=result.identity,
        )

    # Floored in both directions, symmetrically with the CVS gate. A noisy
    # detector must neither fail a clinician on evidence the policy calls
    # absent, nor let a low-confidence major bleed read as a clean pass.
    confident = [e for e in result.bleeding_events if e.confidence >= policy.min_confidence]
    uncertain = [e for e in result.bleeding_events if e.confidence < policy.min_confidence]
    worst = (
        max((e.severity for e in confident), key=lambda severity: BLEEDING_RANK[severity])
        if confident
        else BleedingSeverity.NONE
    )
    if BLEEDING_RANK[worst] >= BLEEDING_RANK[policy.bleeding_fail_at]:
        events = [
            e
            for e in confident
            if BLEEDING_RANK[e.severity] >= BLEEDING_RANK[policy.bleeding_fail_at]
        ]
        return GateResult(
            gate=GateId.BLEEDING,
            status=GateStatus.FAIL,
            reason=f"{worst.value} bleeding was observed",
            perception=result.identity,
            evidence=tuple(f"{e.severity.value}@{e.start_s:.1f}s" for e in events),
        )
    if uncertain:
        return GateResult(
            gate=GateId.BLEEDING,
            status=GateStatus.NOT_ASSESSABLE,
            reason=(
                f"{len(uncertain)} bleeding observation(s) fell below the "
                f"{policy.min_confidence} confidence floor, so severity could "
                f"not be established"
            ),
            perception=result.identity,
            evidence=tuple(f"{e.severity.value}@{e.start_s:.1f}s" for e in uncertain),
        )

    return GateResult(
        gate=GateId.BLEEDING,
        status=GateStatus.PASS,
        reason=(
            f"bleeding was assessed; worst confidently observed severity was "
            f"{worst.value}, below the {policy.bleeding_fail_at.value} threshold"
        ),
        perception=result.identity,
    )


def verify_binding(result: PerceptionResult, episode: Episode) -> None:
    """Confirm ``result`` was computed from ``episode``'s cleared media.

    The de-identification boundary has to be load-bearing here, not only in the
    backend. ``PerceptionResult`` is an ordinary value object: anyone can build
    one by hand, and without this check a hand-built result would be scored
    without any episode, any clearance, or any relationship to real media.
    Carrying the digests is not enough on its own -- they have to be checked
    against something.

    Raises:
        DeidentificationBoundaryError: If the episode is not cleared, or if the
            result's media or attestation digests do not match it.
    """
    episode.require_readable()
    cleared = episode.readable_media
    video = tuple(a for a in cleared if a.kind is MediaKind.ENDOSCOPIC_VIDEO)

    claimed_media = set(result.media_sha256)
    cleared_media = {a.sha256 for a in cleared}
    required_video = {a.sha256 for a in video}

    # Subset on one side, superset on the other. Every claimed digest must be
    # cleared media of this episode, so nothing foreign can be scored; and
    # every endoscopic video must be claimed, so no video can be quietly left
    # out of the material a verdict is said to rest on. Subset rather than
    # equality on the first half lets a backend that also reads kinematics
    # bind it without loosening anything.
    unknown = claimed_media - cleared_media
    if unknown:
        msg = (
            f"perception result from {result.identity} names media digests that are "
            f"not cleared media of episode {episode.id}; a result may only be scored "
            f"against the episode it was computed from"
        )
        raise DeidentificationBoundaryError(msg)
    if not required_video <= claimed_media:
        msg = (
            f"perception result from {result.identity} omits endoscopic video of "
            f"episode {episode.id}; a verdict must rest on all of the video it "
            f"claims to assess"
        )
        raise DeidentificationBoundaryError(msg)

    claimed_attestations = set(result.deid_attestation_sha256)
    expected_attestations = {
        a.deid_attestation_sha256
        for a in cleared
        if a.sha256 in claimed_media and a.deid_attestation_sha256 is not None
    }
    if claimed_attestations != expected_attestations:
        msg = (
            f"perception result from {result.identity} names de-identification "
            f"attestations that do not match the media it claims for episode "
            f"{episode.id}"
        )
        raise DeidentificationBoundaryError(msg)


def evaluate_all(
    result: PerceptionResult,
    episode: Episode,
    procedure: Procedure,
    policy: GatePolicy | None = None,
) -> SafetyGateSet:
    """Evaluate every hard gate for one episode.

    Takes the episode as well as the result so the de-identification binding
    can be verified rather than assumed.

    Raises:
        DeidentificationBoundaryError: If ``result`` is not bound to
            ``episode``'s cleared media.
    """
    verify_binding(result, episode)
    active = policy or GatePolicy()
    return SafetyGateSet(
        results=(
            evaluate_cvs(result, procedure, active),
            evaluate_proximity(result, active),
            evaluate_bleeding(result, active),
        )
    )
