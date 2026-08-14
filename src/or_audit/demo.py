"""A synthetic end-to-end run, for the CLI and for the end-to-end test.

Everything here is generated. There is no real surgical video in this
repository and there must not be: PLAN.md section 8 and V-3 are unresolved, so
the project holds no clinical media at all. The synthetic frames are built to
have the pixel statistics the detectors key on -- red-dominated in-body imagery,
broad-spectrum room imagery, a static overlay block -- which is enough to
exercise the whole chain honestly.

What the demo proves: the layers compose, the audit chain verifies end to end,
the de-identification gate holds, and the artifacts are producible. What it
does not prove: that any detector works on real footage. Nothing here should be
cited as evidence of clinical performance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from or_audit.audit.trail import Actor, ActorKind, AuditTrail
from or_audit.decision.rule import DecisionRule
from or_audit.deid.pipeline import analyze, redact
from or_audit.deid.policy import DeidPolicy
from or_audit.deid.writer import NpzFrameWriter
from or_audit.domain.entities import Episode, Institution, Procedure, Surgeon
from or_audit.domain.enums import Jurisdiction, SkillBand, ThresholdOwner
from or_audit.domain.ids import (
    new_institution_id,
    new_procedure_id,
    new_surgeon_id,
    new_system_id,
)
from or_audit.ingest.manifest import EpisodeManifest, MediaManifest, ingest_episode
from or_audit.media.frames import InMemoryFrameSource, NpzFrameSource
from or_audit.perception.backend import AnnotationBackend
from or_audit.perception.observations import (
    CvsCriterion,
    CvsObservation,
    ObservationKind,
    PhaseSegment,
    SurgicalPhase,
)
from or_audit.pipeline import AssessmentResult, assess_episode
from or_audit.report import CredentialingReport, build_learning_curve
from or_audit.scoring.skill import (
    GearsDomain,
    GearsRating,
    ProficiencyItem,
    ProficiencyResult,
    SkillScore,
)

FRAME_RATE = 30.0
IN_BODY_FRAMES = 240
EXIT_FRAMES = 30
HEIGHT, WIDTH = 64, 96


def in_body_frame(seed: int) -> np.ndarray:
    """Red-dominated imagery, as endoscopic video is."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[..., 0] = rng.integers(180, 215, (HEIGHT, WIDTH))
    frame[..., 1] = rng.integers(35, 70, (HEIGHT, WIDTH))
    frame[..., 2] = rng.integers(35, 70, (HEIGHT, WIDTH))
    return frame


def room_frame(seed: int) -> np.ndarray:
    """Broad-spectrum imagery, as the operating room is."""
    rng = np.random.default_rng(seed + 10_000)
    return rng.integers(110, 160, (HEIGHT, WIDTH, 3), dtype=np.uint8).astype(np.uint8)


def synthetic_source() -> InMemoryFrameSource:
    """An in-body recording with a burned-in overlay and a closing exit."""
    frames = []
    for index in range(IN_BODY_FRAMES + EXIT_FRAMES):
        frame = in_body_frame(index) if index < IN_BODY_FRAMES else room_frame(index)
        # A static block standing in for burned-in patient identifiers.
        frame[0:16, 0:40] = 255
        frames.append(frame)
    return InMemoryFrameSource(frames, frame_rate=FRAME_RATE)


@dataclass(frozen=True)
class DemoOutcome:
    """Everything a demo run produced."""

    trail: AuditTrail
    assessments: tuple[AssessmentResult, ...]
    report: CredentialingReport
    deid_frames_dropped: int
    deid_boxes_masked: int


def _fixed_clock(moment: datetime) -> Callable[[], datetime]:
    """A clock returning one instant, so demo runs are reproducible."""

    def clock() -> datetime:
        return moment

    return clock


def _skill(rater: str, met_items: int) -> SkillScore:
    items = list(ProficiencyItem)
    return SkillScore(
        band_at_episode=SkillBand.ATTENDING,
        rater=rater,
        proficiency=tuple(
            ProficiencyResult(item=item, met=index < met_items) for index, item in enumerate(items)
        ),
        gears=tuple(GearsRating(domain=domain, score=4) for domain in GearsDomain),
    )


def run_demo(workdir: Path, *, episodes: int = 8) -> DemoOutcome:
    """Run the full chain on synthetic data.

    Args:
        workdir: Directory for redacted output.
        episodes: How many episodes to assess.

    Returns:
        The audit trail, the per-episode results and the credentialing report.
    """
    trail = AuditTrail()
    ingest_actor = Actor(kind=ActorKind.CUSTOMER_SYSTEM, ref="demo-import")
    deid_actor = Actor(kind=ActorKind.SERVICE, ref="deid-pipeline")
    score_actor = Actor(kind=ActorKind.SERVICE, ref="scoring-pipeline")

    institution = Institution(
        id=new_institution_id(),
        display_name="Demo Health System",
        jurisdiction=Jurisdiction.US_STATE,
        # Left unconfirmed on purpose, so the demo exercises the section 9
        # attribution-withheld path rather than the permissive one.
        peer_review_protection_confirmed=False,
    )
    procedure = Procedure(
        id=new_procedure_id(),
        code="CHOLE-ROB",
        display_name="Robotic cholecystectomy",
        cvs_applicable=True,
    )
    surgeon = Surgeon(
        id=new_surgeon_id(),
        institution_id=institution.id,
        external_ref="attending-0041",
        band=SkillBand.ATTENDING,
    )
    system_id = new_system_id()
    rule = DecisionRule(
        version="1",
        threshold_owner=ThresholdOwner.CUSTOMER,
        threshold_provenance="Demo credentialing committee minute 2026-02-11",
    )
    policy = DeidPolicy(
        # Truthful for this data and only this data: the synthetic frames are
        # generated below with a 16-pixel-tall overlay, so the recall bound is
        # measured rather than assumed. A real deployment must substitute its
        # own capture survey (PLAN.md V-10); it may not reuse this string.
        overlay_bound_validated_against=(
            "synthetic demo frames: overlay generated at 16px tall by "
            "or_audit.demo.synthetic_source, versus an 8px recall bound"
        )
    )

    results: list[AssessmentResult] = []
    dropped = 0
    masked = 0

    for index in range(episodes):
        source = synthetic_source()
        performed_at = datetime(2026, 1, 6, 8, 0, tzinfo=UTC) + timedelta(days=7 * index)

        manifest = EpisodeManifest(
            institution_id=institution.id,
            procedure_id=procedure.id,
            surgeon_id=surgeon.id,
            system_id=system_id,
            band_at_episode=SkillBand.ATTENDING,
            performed_at=performed_at,
            external_ref=f"DEMO-CASE-{index + 1:03d}",
            media=(
                MediaManifest(
                    kind="endoscopic_video",
                    uri=f"memory://demo/case-{index + 1}.raw",
                    sha256=f"{index:064x}",
                    duration_seconds=source.frame_count / FRAME_RATE,
                    frame_rate=FRAME_RATE,
                ),
            ),
        )
        raw_episode = ingest_episode(manifest, trail=trail, actor=ingest_actor)

        analysed, plan = analyze(
            raw_episode.media[0], source, policy, trail=trail, actor=deid_actor
        )
        cleared_asset, _ = redact(
            analysed,
            source,
            plan,
            policy,
            NpzFrameWriter(workdir / f"case-{index + 1}.npz"),
            performed_by="deid-pipeline",
            trail=trail,
            actor=deid_actor,
        )
        dropped += sum(1 for _ in plan.dropped_segments)
        masked += len(plan.masked_boxes)

        redacted = NpzFrameSource(workdir / f"case-{index + 1}.npz")
        episode = Episode(
            **{
                **raw_episode.model_dump(mode="python"),
                "media": (
                    cleared_asset.model_copy(
                        update={"duration_seconds": redacted.frame_count / FRAME_RATE}
                    ),
                ),
            }
        )

        # Annotations: the critical view achieved before clipping, improving
        # proficiency over the series so the learning curve has a direction.
        clip_start = (redacted.frame_count / FRAME_RATE) * 0.7
        backend = AnnotationBackend(
            observes=frozenset(ObservationKind),
            phases=(
                PhaseSegment(
                    phase=SurgicalPhase.CALOT_TRIANGLE_DISSECTION,
                    start_s=0.5,
                    end_s=clip_start,
                    confidence=0.95,
                ),
                PhaseSegment(
                    phase=SurgicalPhase.CLIPPING_AND_CUTTING,
                    start_s=clip_start,
                    end_s=redacted.frame_count / FRAME_RATE,
                    confidence=0.95,
                ),
            ),
            cvs=tuple(
                CvsObservation(
                    criterion=criterion,
                    achieved=True,
                    at_s=clip_start * 0.8,
                    confidence=0.92,
                )
                for criterion in CvsCriterion
            ),
        )

        results.append(
            assess_episode(
                episode=episode,
                institution=institution,
                procedure=procedure,
                surgeon=surgeon,
                backend=backend,
                skill=_skill("rater-0041", 5 if index < episodes // 2 else 7),
                rule=rule,
                trail=trail,
                actor=score_actor,
                clock=_fixed_clock(performed_at + timedelta(days=1)),
            )
        )

    scorecards = tuple(r.scorecard for r in results)
    report = CredentialingReport(
        institution=institution.display_name,
        surgeon_ref=scorecards[0].surgeon_ref,
        band=SkillBand.ATTENDING,
        generated_at=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        rule_description=rule.describe(),
        scorecards=scorecards,
        curve=build_learning_curve(scorecards, band=SkillBand.ATTENDING),
        attribution_note=(
            None
            if institution.peer_review_protection_confirmed
            else (
                "Individual attribution withheld: this institution's peer-review "
                "protection posture is unconfirmed (PLAN.md section 9, V-3)."
            )
        ),
    )
    return DemoOutcome(
        trail=trail,
        assessments=tuple(results),
        report=report,
        deid_frames_dropped=dropped,
        deid_boxes_masked=masked,
    )
