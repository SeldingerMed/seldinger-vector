"""End-to-end assessment of one episode.

Wires the layers in the order PLAN.md section 7.1 specifies, and enforces the
ordering rather than trusting the caller to follow it:

    ingest -> de-identify -> perceive -> hard gates -> skill -> determination

Every transition appends to the audit trail, so the artifact at the end is
accompanied by a tamper-evident record of what produced it. The pipeline is a
function rather than a class because it holds no state worth keeping: the
episode, the trail, and the artifacts are the state, and hiding them inside an
object would make the ordering less visible, not more.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from or_audit.audit.trail import Actor, AuditAction, AuditTrail
from or_audit.decision.record import DecisionRecord, RaterDisagreement
from or_audit.decision.rule import DecisionRule
from or_audit.deid.attestation import DeidAttestation
from or_audit.domain.entities import Episode, Institution, Procedure, Surgeon
from or_audit.domain.ids import new_decision_id
from or_audit.errors import DeidentificationBoundaryError
from or_audit.perception.backend import PerceptionBackend
from or_audit.perception.observations import PerceptionResult
from or_audit.report import EpisodeScorecard, build_scorecard
from or_audit.scoring.gates import GatePolicy, SafetyGateSet, evaluate_all
from or_audit.scoring.skill import ScoreVector, SkillScore


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class AssessmentResult:
    """Everything one pass over an episode produced."""

    episode: Episode
    perception: PerceptionResult
    gates: SafetyGateSet
    vector: ScoreVector
    decision: DecisionRecord
    scorecard: EpisodeScorecard

    def render(self) -> str:
        """The scorecard as text."""
        return self.scorecard.render()


def assess_episode(
    *,
    episode: Episode,
    institution: Institution,
    procedure: Procedure,
    surgeon: Surgeon,
    backend: PerceptionBackend,
    skill: SkillScore,
    rule: DecisionRule,
    trail: AuditTrail,
    actor: Actor,
    gate_policy: GatePolicy | None = None,
    disagreements: tuple[RaterDisagreement, ...] = (),
    decided_by: str = "svc-decision",
    clock: Callable[[], datetime] = _utc_now,
) -> AssessmentResult:
    """Assess one de-identified episode end to end.

    Args:
        episode: The case. Must be fully de-identified; this is checked twice,
            once by the backend and once by the gate binding, and neither check
            is redundant -- they close different holes.
        institution: Owner of the episode; governs attribution (section 9).
        procedure: Determines which gates apply.
        surgeon: Subject of the assessment.
        backend: Perception backend.
        skill: Expert or automated skill scores for this episode.
        rule: The pre-registered decision rule.
        trail: Audit trail; every transition is appended.
        actor: Principal responsible for this run.
        gate_policy: Gate thresholds; defaults apply if omitted.
        disagreements: Points where the panel split, surfaced on the record.
        decided_by: Principal recorded as issuing the determination.
        clock: Injectable time source.

    Returns:
        The perception result, gates, vector, decision record and scorecard.

    Raises:
        DeidentificationBoundaryError: If the episode is not cleared, or the
            perception result is not bound to it.
    """
    policy = gate_policy or GatePolicy()

    # Fail fast and loudly. The layers below check this too, but a caller who
    # reaches the pipeline with raw media should learn it here rather than three
    # frames deep in a detector.
    episode.require_readable()

    perception = backend.analyse(episode)
    trail.append(
        actor=actor,
        action=AuditAction.PERCEPTION_COMPLETED,
        subject_ref=episode.id,
        payload={
            "backend": perception.identity,
            "observes": sorted(k.value for k in perception.observes),
            "duration_s": perception.duration_s,
        },
    )

    gates = evaluate_all(perception, episode, procedure, policy)
    for gate in gates.results:
        trail.append(
            actor=actor,
            action=AuditAction.SAFETY_GATE_EVALUATED,
            subject_ref=episode.id,
            payload={
                "gate": gate.gate.value,
                "status": gate.status.value,
                "reason": gate.reason,
            },
        )

    vector = ScoreVector(gates=gates, skill=skill)
    trail.append(
        actor=actor,
        action=AuditAction.SCORE_COMPUTED,
        subject_ref=episode.id,
        payload={
            "rater": skill.rater,
            "proficiency_met": skill.met_count,
            "proficiency_assessable": len(skill.assessable),
            "gears_total": skill.gears_total,
        },
    )

    determination, reason = rule.apply(vector)
    decision = DecisionRecord(
        id=new_decision_id(),
        episode_id=episode.id,
        surgeon_id=surgeon.id,
        determination=determination,
        reason=reason,
        decided_at=clock(),
        decided_by=decided_by,
        rule_version=rule.version,
        perception_identity=perception.identity,
        gate_policy_version=policy.version,
        disagreements=disagreements,
    )
    trail.append(
        actor=actor,
        action=AuditAction.DECISION_ISSUED,
        subject_ref=episode.id,
        payload={
            "decision_id": decision.id,
            "determination": determination.value,
            "rule_version": rule.version,
            "decision_sha256": decision.digest,
            "panel_disagreements": len(disagreements),
        },
    )

    scorecard = build_scorecard(
        episode_id=episode.id,
        performed_at=episode.performed_at,
        procedure=procedure,
        band=episode.band_at_episode,
        surgeon=surgeon,
        institution=institution,
        gates=gates.results,
        skill=skill,
        decision=decision,
    )
    return AssessmentResult(
        episode=episode,
        perception=perception,
        gates=gates,
        vector=vector,
        decision=decision,
        scorecard=scorecard,
    )


def require_cleared(episode: Episode, attestations: tuple[DeidAttestation, ...]) -> None:
    """Confirm every surviving asset has a matching attestation.

    A belt-and-braces check for callers assembling an episode from stored
    attestations rather than from a live de-identification run. It catches the
    case where an asset is marked attested but the attestation it points at is
    missing from the set the caller holds.

    Raises:
        DeidentificationBoundaryError: On any asset without a matching
            attestation digest.
    """
    episode.require_readable()
    held = {a.digest for a in attestations}
    for asset in episode.media:
        if not asset.is_present:
            continue
        if asset.deid_attestation_sha256 not in held:
            msg = (
                f"media {asset.id} claims attestation "
                f"{asset.deid_attestation_sha256} but no such attestation was "
                f"supplied; the clearance cannot be evidenced"
            )
            raise DeidentificationBoundaryError(msg)
