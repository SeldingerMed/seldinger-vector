"""Hard safety gates: PLAN.md section 7.1.

The gates exist to produce verdicts that survive challenge, so the tests are
organised around the ways a verdict can be wrong: clearing something it should
not, failing something it should not, and -- most importantly -- reporting a
pass when it simply could not see.
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
from or_audit.errors import ScoreContractError
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
    GateId,
    GatePolicy,
    evaluate_all,
    evaluate_bleeding,
    evaluate_cvs,
    evaluate_proximity,
)

POLICY = GatePolicy()

MEDIA_SHA = "a" * 64
ATTESTATION_SHA = "d" * 64
ALL_KINDS = frozenset(ObservationKind)


@pytest.fixture
def episode() -> Episode:
    """An episode whose cleared media matches the digests the fixtures use."""
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
                sha256=MEDIA_SHA,
                duration_seconds=1800.0,
                frame_rate=30.0,
                deid_status=DeidStatus.ATTESTED,
                deid_attestation_sha256=ATTESTATION_SHA,
            ),
        ),
    )


@pytest.fixture
def chole() -> Procedure:
    return Procedure(
        id=new_procedure_id(),
        code="CHOLE-ROB",
        display_name="Robotic cholecystectomy",
        cvs_applicable=True,
    )


@pytest.fixture
def prostatectomy() -> Procedure:
    return Procedure(
        id=new_procedure_id(),
        code="PROST-ROB",
        display_name="Robotic prostatectomy",
        cvs_applicable=False,
    )


def result(**overrides: object) -> PerceptionResult:
    """A perception result with everything clean unless overridden."""
    base: dict[str, object] = {
        "backend": "expert-annotation",
        "backend_version": "1",
        "media_sha256": (MEDIA_SHA,),
        "deid_attestation_sha256": (ATTESTATION_SHA,),
        "observes": ALL_KINDS,
        "duration_s": 1800.0,
        "phases": (
            PhaseSegment(
                phase=SurgicalPhase.CALOT_TRIANGLE_DISSECTION,
                start_s=100.0,
                end_s=600.0,
                confidence=0.95,
            ),
            PhaseSegment(
                phase=SurgicalPhase.CLIPPING_AND_CUTTING,
                start_s=600.0,
                end_s=750.0,
                confidence=0.95,
            ),
        ),
        "cvs": tuple(
            CvsObservation(criterion=c, achieved=True, at_s=500.0, confidence=0.9)
            for c in CvsCriterion
        ),
    }
    return PerceptionResult(**(base | overrides))


class TestCvsGate:
    def test_all_three_criteria_before_clipping_passes(self, chole):
        gate = evaluate_cvs(result(), chole, POLICY)
        assert gate.status is GateStatus.PASS
        assert gate.is_clear

    def test_one_criterion_not_achieved_fails(self, chole):
        cvs = (
            CvsObservation(criterion=CvsCriterion.TRIANGLE_CLEARED, achieved=False, confidence=0.9),
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
        gate = evaluate_cvs(result(cvs=cvs), chole, POLICY)
        assert gate.status is GateStatus.FAIL
        assert "hepatocystic_triangle_cleared" in gate.reason

    def test_two_of_three_is_a_fail_not_a_partial_pass(self, chole):
        """There is no partial credit on a safety gate."""
        cvs = (
            CvsObservation(
                criterion=CvsCriterion.TRIANGLE_CLEARED,
                achieved=True,
                at_s=500.0,
                confidence=0.9,
            ),
            CvsObservation(
                criterion=CvsCriterion.CYSTIC_PLATE_EXPOSED,
                achieved=True,
                at_s=500.0,
                confidence=0.9,
            ),
            CvsObservation(
                criterion=CvsCriterion.TWO_STRUCTURES_ONLY, achieved=False, confidence=0.9
            ),
        )
        assert evaluate_cvs(result(cvs=cvs), chole, POLICY).status is GateStatus.FAIL

    def test_criteria_achieved_after_clipping_fails(self, chole):
        """The view must precede the irreversible step, not follow it."""
        cvs = tuple(
            CvsObservation(criterion=c, achieved=True, at_s=700.0, confidence=0.9)
            for c in CvsCriterion
        )
        gate = evaluate_cvs(result(cvs=cvs), chole, POLICY)
        assert gate.status is GateStatus.FAIL
        assert "strictly before" in gate.reason

    def test_last_criterion_is_the_one_that_counts(self, chole):
        """Completion time is the max, not the min: all three must precede."""
        cvs = (
            CvsObservation(
                criterion=CvsCriterion.TRIANGLE_CLEARED,
                achieved=True,
                at_s=200.0,
                confidence=0.9,
            ),
            CvsObservation(
                criterion=CvsCriterion.CYSTIC_PLATE_EXPOSED,
                achieved=True,
                at_s=300.0,
                confidence=0.9,
            ),
            CvsObservation(
                criterion=CvsCriterion.TWO_STRUCTURES_ONLY,
                achieved=True,
                at_s=650.0,
                confidence=0.9,
            ),
        )
        assert evaluate_cvs(result(cvs=cvs), chole, POLICY).status is GateStatus.FAIL


class TestCvsCannotClearWhatItCannotSee:
    """The failure mode that matters most: silently passing on no evidence."""

    def test_missing_criterion_is_unassessable_not_pass(self, chole):
        cvs = tuple(
            CvsObservation(criterion=c, achieved=True, at_s=500.0, confidence=0.9)
            for c in list(CvsCriterion)[:2]
        )
        gate = evaluate_cvs(result(cvs=cvs), chole, POLICY)
        assert gate.status is GateStatus.NOT_ASSESSABLE
        assert not gate.is_clear

    def test_unknown_criterion_is_unassessable_not_fail(self, chole):
        """Three-valued: 'could not tell' is not a finding against the surgeon."""
        cvs = (
            CvsObservation(criterion=CvsCriterion.TRIANGLE_CLEARED, achieved=None, confidence=0.0),
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
        assert evaluate_cvs(result(cvs=cvs), chole, POLICY).status is GateStatus.NOT_ASSESSABLE

    def test_low_confidence_achievement_does_not_clear(self, chole):
        cvs = tuple(
            CvsObservation(criterion=c, achieved=True, at_s=500.0, confidence=0.3)
            for c in CvsCriterion
        )
        gate = evaluate_cvs(result(cvs=cvs), chole, POLICY)
        assert gate.status is GateStatus.NOT_ASSESSABLE
        assert "confidence below" in gate.reason

    def test_explicit_failure_outranks_missing_evidence(self, chole):
        """A known violation is reported as one, not hidden behind a gap."""
        cvs = (
            CvsObservation(criterion=CvsCriterion.TRIANGLE_CLEARED, achieved=False, confidence=0.9),
            CvsObservation(
                criterion=CvsCriterion.CYSTIC_PLATE_EXPOSED, achieved=None, confidence=0.0
            ),
        )
        assert evaluate_cvs(result(cvs=cvs), chole, POLICY).status is GateStatus.FAIL

    def test_unobserved_clipping_phase_blocks_a_pass(self, chole):
        """Criteria met, but the timing claim is uncheckable."""
        phases = (
            PhaseSegment(
                phase=SurgicalPhase.CALOT_TRIANGLE_DISSECTION,
                start_s=100.0,
                end_s=600.0,
                confidence=0.95,
            ),
        )
        gate = evaluate_cvs(result(phases=phases), chole, POLICY)
        assert gate.status is GateStatus.NOT_ASSESSABLE
        assert "never observed" in gate.reason

    def test_no_observations_at_all_is_unassessable(self, chole):
        assert evaluate_cvs(result(cvs=()), chole, POLICY).status is GateStatus.NOT_ASSESSABLE


class TestCvsApplicability:
    def test_non_biliary_procedure_is_not_applicable(self, prostatectomy):
        gate = evaluate_cvs(result(), prostatectomy, POLICY)
        assert gate.status is GateStatus.NOT_APPLICABLE

    def test_not_applicable_is_not_a_pass(self, prostatectomy):
        """Otherwise every non-biliary case inflates apparent safety coverage."""
        assert not evaluate_cvs(result(), prostatectomy, POLICY).is_clear


class TestProximityGate:
    def test_no_events_passes(self):
        assert evaluate_proximity(result(), POLICY).status is GateStatus.PASS

    def test_close_approach_to_the_bile_duct_fails(self):
        events = (
            ProximityEvent(
                structure=CriticalStructure.COMMON_BILE_DUCT,
                at_s=400.0,
                distance_mm=2.0,
                confidence=0.9,
            ),
        )
        gate = evaluate_proximity(result(proximity_events=events), POLICY)
        assert gate.status is GateStatus.FAIL

    def test_distant_approach_passes(self):
        events = (
            ProximityEvent(
                structure=CriticalStructure.COMMON_BILE_DUCT,
                at_s=400.0,
                distance_mm=25.0,
                confidence=0.9,
            ),
        )
        assert evaluate_proximity(result(proximity_events=events), POLICY).status is GateStatus.PASS

    def test_unmeasured_distance_is_unassessable_not_safe(self):
        """'We could not measure it' and 'it was far away' are different claims."""
        events = (
            ProximityEvent(
                structure=CriticalStructure.COMMON_BILE_DUCT,
                at_s=400.0,
                distance_mm=None,
                confidence=0.9,
            ),
        )
        gate = evaluate_proximity(result(proximity_events=events), POLICY)
        assert gate.status is GateStatus.NOT_ASSESSABLE

    def test_a_real_alarm_outranks_an_unmeasured_event(self):
        events = (
            ProximityEvent(
                structure=CriticalStructure.COMMON_BILE_DUCT,
                at_s=400.0,
                distance_mm=None,
                confidence=0.9,
            ),
            ProximityEvent(
                structure=CriticalStructure.RIGHT_HEPATIC_ARTERY,
                at_s=420.0,
                distance_mm=1.0,
                confidence=0.9,
            ),
        )
        assert evaluate_proximity(result(proximity_events=events), POLICY).status is GateStatus.FAIL

    def test_low_confidence_alarm_is_unassessable_not_ignored(self):
        """This test previously asserted PASS, which locked in a P0.

        Discarding a 1mm approach to the common bile duct because the detector
        was only 10% sure, then reporting "no instrument approached within the
        alarm distance", is a false statement on an artifact that can be
        adverse to a clinician. The gate cannot see clearly, so it must not
        clear.
        """
        events = (
            ProximityEvent(
                structure=CriticalStructure.COMMON_BILE_DUCT,
                at_s=400.0,
                distance_mm=1.0,
                confidence=0.1,
            ),
        )
        gate = evaluate_proximity(result(proximity_events=events), POLICY)
        assert gate.status is GateStatus.NOT_ASSESSABLE
        assert not gate.is_clear

    def test_approach_to_a_non_critical_structure_is_ignored(self):
        """Touching the cystic duct is the operation, not a near-miss."""
        events = (
            ProximityEvent(
                structure=CriticalStructure.CYSTIC_DUCT,
                at_s=400.0,
                distance_mm=0.5,
                confidence=0.99,
            ),
        )
        assert evaluate_proximity(result(proximity_events=events), POLICY).status is GateStatus.PASS

    def test_boundary_distance_is_inclusive(self):
        events = (
            ProximityEvent(
                structure=CriticalStructure.COMMON_BILE_DUCT,
                at_s=400.0,
                distance_mm=POLICY.proximity_alarm_mm,
                confidence=0.9,
            ),
        )
        assert evaluate_proximity(result(proximity_events=events), POLICY).status is GateStatus.FAIL


class TestBleedingGate:
    def test_no_bleeding_passes(self):
        assert evaluate_bleeding(result(), POLICY).status is GateStatus.PASS

    @pytest.mark.parametrize(
        ("severity", "expected"),
        [
            (BleedingSeverity.MINOR, GateStatus.PASS),
            (BleedingSeverity.MODERATE, GateStatus.PASS),
            (BleedingSeverity.MAJOR, GateStatus.FAIL),
        ],
    )
    def test_severity_threshold(self, severity, expected):
        events = (BleedingEvent(severity=severity, start_s=300.0, end_s=320.0, confidence=0.9),)
        assert evaluate_bleeding(result(bleeding_events=events), POLICY).status is expected

    def test_worst_event_governs(self):
        events = (
            BleedingEvent(
                severity=BleedingSeverity.MINOR, start_s=100.0, end_s=110.0, confidence=0.9
            ),
            BleedingEvent(
                severity=BleedingSeverity.MAJOR, start_s=300.0, end_s=340.0, confidence=0.9
            ),
        )
        assert evaluate_bleeding(result(bleeding_events=events), POLICY).status is GateStatus.FAIL

    def test_threshold_is_configurable(self):
        events = (
            BleedingEvent(
                severity=BleedingSeverity.MODERATE, start_s=300.0, end_s=320.0, confidence=0.9
            ),
        )
        strict = GatePolicy(bleeding_fail_at=BleedingSeverity.MODERATE)
        assert evaluate_bleeding(result(bleeding_events=events), strict).status is GateStatus.FAIL


class TestGateSetRefusesToCollapse:
    """Section 7.1: hard gates never average into a score."""

    def test_float_raises(self, episode, chole):
        gates = evaluate_all(result(), episode, chole)
        with pytest.raises(ScoreContractError, match="no scalar value"):
            float(gates)

    def test_int_raises(self, episode, chole):
        gates = evaluate_all(result(), episode, chole)
        with pytest.raises(ScoreContractError, match="no scalar value"):
            int(gates)

    def test_bool_raises_so_the_question_must_be_named(self, episode, chole):
        """`if gates:` reads as 'did it pass' but would mean 'is it non-empty'."""
        gates = evaluate_all(result(), episode, chole)
        with pytest.raises(ScoreContractError, match="no truth value"):
            bool(gates)

    def test_sum_raises(self, episode, chole):
        """Calls sum() rather than re-testing float(), which the old test did."""
        gates = evaluate_all(result(), episode, chole)
        with pytest.raises(TypeError):
            sum(gates)  # type: ignore[arg-type]

    def test_iteration_yields_gate_results_not_field_tuples(self, episode, chole):
        """pydantic's default __iter__ would yield ("results", ...) pairs."""
        gates = evaluate_all(result(), episode, chole)
        assert [g.gate for g in gates] == [r.gate for r in gates.results]


class TestGateSetQueries:
    def test_clean_episode_is_all_clear(self, episode, chole):
        gates = evaluate_all(result(), episode, chole)
        assert gates.all_clear
        assert not gates.any_failed

    def test_unassessable_gate_blocks_all_clear(self, episode, chole):
        """An episode nobody could evaluate has not been cleared."""
        gates = evaluate_all(result(cvs=()), episode, chole)
        assert not gates.all_clear
        assert not gates.any_failed
        assert len(gates.unassessable) == 1

    def test_not_applicable_gate_blocks_all_clear(self, episode, prostatectomy):
        assert not evaluate_all(result(), episode, prostatectomy).all_clear

    def test_failed_gate_is_reported(self, episode, chole):
        events = (
            BleedingEvent(
                severity=BleedingSeverity.MAJOR, start_s=300.0, end_s=340.0, confidence=0.9
            ),
        )
        gates = evaluate_all(result(bleeding_events=events), episode, chole)
        assert gates.any_failed
        assert gates.failed[0].gate is GateId.BLEEDING

    def test_every_gate_is_retrievable_by_id(self, episode, chole):
        gates = evaluate_all(result(), episode, chole)
        assert len(gates) == 3
        for gate_id in GateId:
            assert gates.get(gate_id) is not None

    def test_every_result_carries_a_reason_and_backend(self, episode, chole):
        for gate in evaluate_all(result(), episode, chole).results:
            assert gate.reason
            assert gate.perception == "expert-annotation@1"
