"""Regression tests for review findings on the hard-gate layer.

The layer makes one central promise, in its own module docstring: *a gate that
cannot see cannot clear*. Review found that only the CVS gate's positive branch
actually kept it. These tests pin the promise for every gate and every
direction, because the failures were all of the same shape -- evidence quietly
discarded, then absence of evidence reported as absence of a problem.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from or_audit.domain.entities import Episode, MediaAsset, Procedure
from or_audit.domain.enums import DeidStatus, GateStatus, MediaKind, SkillBand
from or_audit.domain.ids import (
    new_episode_id,
    new_institution_id,
    new_media_asset_id,
    new_procedure_id,
    new_surgeon_id,
    new_system_id,
)
from or_audit.errors import DeidentificationBoundaryError, DomainInvariantError
from or_audit.perception.backend import AnnotationBackend
from or_audit.perception.observations import (
    BleedingEvent,
    BleedingSeverity,
    CriticalStructure,
    CvsCriterion,
    CvsObservation,
    ObservationKind,
    PerceptionResult,
    PhaseSegment,
    ProximityEvent,
    SurgicalPhase,
)
from or_audit.scoring.gates import (
    GatePolicy,
    evaluate_all,
    evaluate_bleeding,
    evaluate_cvs,
    evaluate_proximity,
    verify_binding,
)

POLICY = GatePolicy()
MEDIA_SHA = "a" * 64
ATTESTATION_SHA = "d" * 64


def episode_for(media_sha: str = MEDIA_SHA, attestation: str = ATTESTATION_SHA) -> Episode:
    episode_id = new_episode_id()
    return Episode(
        id=episode_id,
        institution_id=new_institution_id(),
        procedure_id=new_procedure_id(),
        surgeon_id=new_surgeon_id(),
        system_id=new_system_id(),
        band_at_episode=SkillBand.ATTENDING,
        performed_at=datetime(2026, 3, 4, 14, 30, tzinfo=UTC),
        media=(
            MediaAsset(
                id=new_media_asset_id(),
                episode_id=episode_id,
                kind=MediaKind.ENDOSCOPIC_VIDEO,
                raw_uri="file:///deid/case.npz",
                sha256=media_sha,
                duration_seconds=1800.0,
                frame_rate=30.0,
                deid_status=DeidStatus.ATTESTED,
                deid_attestation_sha256=attestation,
            ),
        ),
    )


def result(**overrides: object) -> PerceptionResult:
    base: dict[str, object] = {
        "backend": "expert-annotation",
        "backend_version": "1",
        "media_sha256": (MEDIA_SHA,),
        "deid_attestation_sha256": (ATTESTATION_SHA,),
        "observes": frozenset(ObservationKind),
        "duration_s": 1800.0,
    }
    return PerceptionResult(**(base | overrides))


@pytest.fixture
def chole() -> Procedure:
    return Procedure(
        id=new_procedure_id(),
        code="CHOLE-ROB",
        display_name="Robotic cholecystectomy",
        cvs_applicable=True,
    )


class TestLowConfidenceNeverClearsAGate:
    """P0: the proximity gate discarded below-floor alarms and reported PASS.

    A 1mm approach to the common bile duct at 50% confidence was dropped, and
    the gate then stated "no instrument approached a critical structure within
    the alarm distance". That sentence was false, on an artifact that can be
    adverse to a named clinician.
    """

    @pytest.mark.parametrize("confidence", [0.0, 0.1, 0.5, 0.59])
    def test_below_floor_alarm_is_unassessable(self, confidence):
        events = (
            ProximityEvent(
                structure=CriticalStructure.COMMON_BILE_DUCT,
                at_s=400.0,
                distance_mm=1.0,
                confidence=confidence,
            ),
        )
        gate = evaluate_proximity(result(proximity_events=events), POLICY)
        assert gate.status is GateStatus.NOT_ASSESSABLE
        assert not gate.is_clear

    def test_at_the_floor_a_real_alarm_still_fails(self):
        events = (
            ProximityEvent(
                structure=CriticalStructure.COMMON_BILE_DUCT,
                at_s=400.0,
                distance_mm=1.0,
                confidence=POLICY.min_confidence,
            ),
        )
        assert evaluate_proximity(result(proximity_events=events), POLICY).status is GateStatus.FAIL

    def test_a_confident_alarm_outranks_an_uncertain_one(self):
        events = (
            ProximityEvent(
                structure=CriticalStructure.COMMON_BILE_DUCT,
                at_s=400.0,
                distance_mm=1.0,
                confidence=0.1,
            ),
            ProximityEvent(
                structure=CriticalStructure.RIGHT_HEPATIC_ARTERY,
                at_s=420.0,
                distance_mm=1.0,
                confidence=0.99,
            ),
        )
        assert evaluate_proximity(result(proximity_events=events), POLICY).status is GateStatus.FAIL

    def test_uncertain_approach_to_a_non_critical_structure_is_still_ignored(self):
        """Touching the cystic duct is the operation, at any confidence."""
        events = (
            ProximityEvent(
                structure=CriticalStructure.CYSTIC_DUCT,
                at_s=400.0,
                distance_mm=0.5,
                confidence=0.2,
            ),
        )
        assert evaluate_proximity(result(proximity_events=events), POLICY).status is GateStatus.PASS


class TestBleedingConfidenceIsSymmetric:
    """P1: bleeding ignored confidence in both directions.

    A major bleed at 1% confidence produced a FAIL -- an adverse finding on
    evidence the policy defines as absent -- while the same policy floored CVS.
    """

    def test_low_confidence_major_bleed_is_unassessable_not_fail(self):
        events = (
            BleedingEvent(
                severity=BleedingSeverity.MAJOR, start_s=300.0, end_s=340.0, confidence=0.01
            ),
        )
        gate = evaluate_bleeding(result(bleeding_events=events), POLICY)
        assert gate.status is GateStatus.NOT_ASSESSABLE

    def test_low_confidence_minor_bleed_is_unassessable_not_pass(self):
        """Symmetry: an unreadable observation is unreadable either way."""
        events = (
            BleedingEvent(
                severity=BleedingSeverity.MINOR, start_s=300.0, end_s=340.0, confidence=0.01
            ),
        )
        assert evaluate_bleeding(result(bleeding_events=events), POLICY).status is (
            GateStatus.NOT_ASSESSABLE
        )

    def test_confident_major_bleed_still_fails(self):
        events = (
            BleedingEvent(
                severity=BleedingSeverity.MAJOR, start_s=300.0, end_s=340.0, confidence=0.9
            ),
        )
        assert evaluate_bleeding(result(bleeding_events=events), POLICY).status is GateStatus.FAIL

    def test_confident_absence_passes(self):
        assert evaluate_bleeding(result(), POLICY).status is GateStatus.PASS


class TestCvsFloorAppliesToNegativeEvidenceToo:
    """P1: a below-floor 'not achieved' produced a FAIL.

    FAIL is the worst verdict available. Resting it on evidence the policy
    defines as absent contradicted the gate's own documented contract.
    """

    def test_low_confidence_not_achieved_is_unassessable(self, chole):
        cvs = (
            CvsObservation(criterion=CvsCriterion.TRIANGLE_CLEARED, achieved=False, confidence=0.1),
            CvsObservation(
                criterion=CvsCriterion.CYSTIC_PLATE_EXPOSED,
                achieved=True,
                at_s=500.0,
                confidence=0.9,
            ),
            CvsObservation(
                criterion=CvsCriterion.TWO_STRUCTURES_ONLY,
                achieved=True,
                at_s=500.0,
                confidence=0.9,
            ),
        )
        phases = (
            PhaseSegment(
                phase=SurgicalPhase.CLIPPING_AND_CUTTING,
                start_s=600.0,
                end_s=700.0,
                confidence=0.95,
            ),
        )
        gate = evaluate_cvs(result(cvs=cvs, phases=phases), chole, POLICY)
        assert gate.status is GateStatus.NOT_ASSESSABLE

    def test_confident_not_achieved_still_fails(self, chole):
        cvs = (
            CvsObservation(criterion=CvsCriterion.TRIANGLE_CLEARED, achieved=False, confidence=0.9),
        )
        assert evaluate_cvs(result(cvs=cvs), chole, POLICY).status is GateStatus.FAIL


class TestPhaseAnchorIsFloored:
    """P2: the clipping phase's own confidence was never checked.

    A phantom 1%-confidence detection could manufacture a timing verdict while
    every other input to the same gate was floored.
    """

    def test_low_confidence_clipping_phase_does_not_anchor_a_verdict(self, chole):
        cvs = tuple(
            CvsObservation(criterion=c, achieved=True, at_s=500.0, confidence=0.9)
            for c in CvsCriterion
        )
        phases = (
            PhaseSegment(
                phase=SurgicalPhase.CLIPPING_AND_CUTTING,
                start_s=100.0,
                end_s=200.0,
                confidence=0.01,
            ),
        )
        gate = evaluate_cvs(result(cvs=cvs, phases=phases), chole, POLICY)
        assert gate.status is GateStatus.NOT_ASSESSABLE
        assert "never observed" in gate.reason


class TestTimingBoundaryIsStrict:
    """nit: completion exactly at the clipping start used to PASS.

    The reason string then read "satisfied by 200.0s, before clipping began at
    200.0s", which is self-contradictory. A view completed at the instant the
    irreversible step begins has not been shown to precede it.
    """

    def test_completion_exactly_at_clipping_start_fails(self, chole):
        cvs = tuple(
            CvsObservation(criterion=c, achieved=True, at_s=600.0, confidence=0.9)
            for c in CvsCriterion
        )
        phases = (
            PhaseSegment(
                phase=SurgicalPhase.CLIPPING_AND_CUTTING,
                start_s=600.0,
                end_s=700.0,
                confidence=0.95,
            ),
        )
        gate = evaluate_cvs(result(cvs=cvs, phases=phases), chole, POLICY)
        assert gate.status is GateStatus.FAIL
        assert "strictly before" in gate.reason

    def test_completion_just_before_passes(self, chole):
        cvs = tuple(
            CvsObservation(criterion=c, achieved=True, at_s=599.9, confidence=0.9)
            for c in CvsCriterion
        )
        phases = (
            PhaseSegment(
                phase=SurgicalPhase.CLIPPING_AND_CUTTING,
                start_s=600.0,
                end_s=700.0,
                confidence=0.95,
            ),
        )
        assert evaluate_cvs(result(cvs=cvs, phases=phases), chole, POLICY).status is (
            GateStatus.PASS
        )


class TestCoverageMustBeDeclared:
    """P1: absent evidence passed, because 'never looked' looked like 'none found'.

    A backend implementing neither proximity nor bleeding detection cleared both
    gates by returning nothing, and the bleeding gate stated "worst observed
    bleeding was none" having observed nothing at all.
    """

    def test_undeclared_proximity_is_unassessable(self):
        out = result(observes=frozenset({ObservationKind.CVS}))
        gate = evaluate_proximity(out, POLICY)
        assert gate.status is GateStatus.NOT_ASSESSABLE
        assert "did not assess" in gate.reason

    def test_undeclared_bleeding_is_unassessable(self):
        out = result(observes=frozenset({ObservationKind.CVS}))
        assert evaluate_bleeding(out, POLICY).status is GateStatus.NOT_ASSESSABLE

    def test_undeclared_cvs_is_unassessable(self, chole):
        out = result(observes=frozenset({ObservationKind.BLEEDING}))
        assert evaluate_cvs(out, chole, POLICY).status is GateStatus.NOT_ASSESSABLE

    def test_declared_coverage_with_no_findings_passes(self):
        """'Measured and found nothing' is a real pass."""
        out = result(observes=frozenset({ObservationKind.BLEEDING}))
        assert evaluate_bleeding(out, POLICY).status is GateStatus.PASS

    def test_reporting_an_undeclared_kind_is_a_contradiction(self):
        with pytest.raises(DomainInvariantError, match="without declaring coverage"):
            result(
                observes=frozenset({ObservationKind.CVS}),
                bleeding_events=(
                    BleedingEvent(
                        severity=BleedingSeverity.MINOR,
                        start_s=1.0,
                        end_s=2.0,
                        confidence=0.9,
                    ),
                ),
            )

    def test_annotation_backend_requires_a_coverage_declaration(self):
        with pytest.raises(DomainInvariantError, match="must declare which observation kinds"):
            AnnotationBackend()


class TestDeidBoundaryIsLoadBearingAtScoringTime:
    """P2: a hand-built result was scored with no episode and no clearance.

    PerceptionResult is an ordinary value object, so carrying digests is not
    enough -- they have to be checked against something.
    """

    def test_matching_result_verifies(self):
        verify_binding(result(), episode_for())

    def test_mismatched_media_digest_is_refused(self, chole):
        out = result(media_sha256=("f" * 64,))
        with pytest.raises(DeidentificationBoundaryError, match="do not match episode"):
            evaluate_all(out, episode_for(), chole)

    def test_mismatched_attestation_digest_is_refused(self, chole):
        out = result(deid_attestation_sha256=("e" * 64,))
        with pytest.raises(DeidentificationBoundaryError, match="attestations that do not match"):
            evaluate_all(out, episode_for(), chole)

    def test_uncleared_episode_is_refused(self, chole):
        episode_id = new_episode_id()
        raw = Episode(
            id=episode_id,
            institution_id=new_institution_id(),
            procedure_id=new_procedure_id(),
            surgeon_id=new_surgeon_id(),
            system_id=new_system_id(),
            band_at_episode=SkillBand.ATTENDING,
            performed_at=datetime(2026, 3, 4, 14, 30, tzinfo=UTC),
            media=(
                MediaAsset(
                    id=new_media_asset_id(),
                    episode_id=episode_id,
                    kind=MediaKind.ENDOSCOPIC_VIDEO,
                    raw_uri="s3://raw/case.mp4",
                    sha256=MEDIA_SHA,
                    duration_seconds=1800.0,
                    deid_status=DeidStatus.RAW,
                ),
            ),
        )
        with pytest.raises(DeidentificationBoundaryError):
            evaluate_all(result(), raw, chole)
