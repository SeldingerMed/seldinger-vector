"""Perception boundary: gating, binding, and observation validity."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from or_audit.domain.entities import Episode, MediaAsset
from or_audit.domain.enums import DeidStatus, MediaKind, SkillBand
from or_audit.domain.ids import (
    new_episode_id,
    new_institution_id,
    new_media_asset_id,
    new_procedure_id,
    new_surgeon_id,
    new_system_id,
)
from or_audit.errors import DeidentificationBoundaryError, DomainInvariantError
from or_audit.perception.backend import AnnotationBackend, PerceptionBackend
from or_audit.perception.observations import (
    BleedingEvent,
    BleedingSeverity,
    CriticalStructure,
    CvsCriterion,
    CvsObservation,
    PerceptionResult,
    PhaseSegment,
    ProximityEvent,
    StructureSighting,
    SurgicalPhase,
)


def build_episode(
    *, deid: DeidStatus = DeidStatus.ATTESTED, duration: float | None = 1800.0
) -> Episode:
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
                sha256="b" * 64,
                duration_seconds=duration,
                frame_rate=30.0,
                deid_status=deid,
                deid_attestation_sha256=("c" * 64 if deid is DeidStatus.ATTESTED else None),
            ),
        ),
    )


class TestDeidentificationGateHolds:
    """Perception must not be a way around section 8."""

    @pytest.mark.parametrize("status", [DeidStatus.RAW, DeidStatus.IN_PROGRESS, DeidStatus.FAILED])
    def test_uncleared_media_cannot_be_perceived(self, status):
        with pytest.raises(DeidentificationBoundaryError):
            AnnotationBackend().analyse(build_episode(deid=status))

    def test_cleared_media_can_be_perceived(self):
        assert AnnotationBackend().analyse(build_episode()).duration_s == 1800.0


class TestResultBinding:
    def test_result_is_bound_to_the_media_digests(self):
        """Otherwise a result could be paired with a different recording."""
        episode = build_episode()
        out = AnnotationBackend().analyse(episode)
        assert out.media_sha256 == (episode.media[0].sha256,)

    def test_result_carries_backend_identity(self):
        assert AnnotationBackend().analyse(build_episode()).identity == "expert-annotation@1"

    def test_backend_satisfies_the_protocol(self):
        assert isinstance(AnnotationBackend(), PerceptionBackend)

    def test_missing_duration_is_refused(self):
        """Timing-dependent gates cannot run against an unknown timeline."""
        with pytest.raises(DomainInvariantError, match="no duration"):
            AnnotationBackend().analyse(build_episode(duration=None))

    def test_result_without_media_digests_is_rejected(self):
        with pytest.raises(DomainInvariantError, match="names no source media"):
            PerceptionResult(backend="x", backend_version="1", media_sha256=(), duration_s=10.0)


class TestAnnotationsMustMatchTheMedia:
    """An annotation past the end of the recording means the annotator was
    working from other material, or the timeline is misaligned."""

    def test_phase_past_the_end_is_refused(self):
        backend = AnnotationBackend(
            phases=(
                PhaseSegment(
                    phase=SurgicalPhase.PREPARATION,
                    start_s=0.0,
                    end_s=9999.0,
                    confidence=0.9,
                ),
            )
        )
        with pytest.raises(DomainInvariantError, match="do not match this media"):
            backend.analyse(build_episode())

    def test_bleeding_past_the_end_is_refused(self):
        backend = AnnotationBackend(
            bleeding_events=(
                BleedingEvent(
                    severity=BleedingSeverity.MINOR,
                    start_s=9000.0,
                    end_s=9999.0,
                    confidence=0.9,
                ),
            )
        )
        with pytest.raises(DomainInvariantError, match="bleeding event ends at"):
            backend.analyse(build_episode())

    def test_proximity_past_the_end_is_refused(self):
        backend = AnnotationBackend(
            proximity_events=(
                ProximityEvent(
                    structure=CriticalStructure.COMMON_BILE_DUCT,
                    at_s=9999.0,
                    distance_mm=1.0,
                    confidence=0.9,
                ),
            )
        )
        with pytest.raises(DomainInvariantError, match="proximity event at"):
            backend.analyse(build_episode())

    def test_structure_past_the_end_is_refused(self):
        backend = AnnotationBackend(
            structures=(
                StructureSighting(
                    structure=CriticalStructure.CYSTIC_DUCT,
                    start_s=9000.0,
                    end_s=9999.0,
                    confidence=0.9,
                ),
            )
        )
        with pytest.raises(DomainInvariantError, match="sighting of"):
            backend.analyse(build_episode())

    def test_cvs_timestamp_past_the_end_is_refused(self):
        backend = AnnotationBackend(
            cvs=(
                CvsObservation(
                    criterion=CvsCriterion.TRIANGLE_CLEARED,
                    achieved=True,
                    at_s=9999.0,
                    confidence=0.9,
                ),
            )
        )
        with pytest.raises(DomainInvariantError, match="recorded at"):
            backend.analyse(build_episode())

    def test_annotations_inside_the_recording_are_accepted(self):
        backend = AnnotationBackend(
            phases=(
                PhaseSegment(
                    phase=SurgicalPhase.PREPARATION,
                    start_s=0.0,
                    end_s=1800.0,
                    confidence=0.9,
                ),
            )
        )
        assert len(backend.analyse(build_episode()).phases) == 1


class TestObservationValidity:
    def test_zero_length_phase_is_rejected(self):
        with pytest.raises(DomainInvariantError, match="positive duration"):
            PhaseSegment(phase=SurgicalPhase.PREPARATION, start_s=5.0, end_s=5.0, confidence=0.9)

    def test_reversed_structure_span_is_rejected(self):
        with pytest.raises(DomainInvariantError, match="positive duration"):
            StructureSighting(
                structure=CriticalStructure.CYSTIC_DUCT,
                start_s=10.0,
                end_s=5.0,
                confidence=0.9,
            )

    def test_bleeding_event_cannot_be_severity_none(self):
        """A non-event should be omitted, not recorded as an event."""
        with pytest.raises(DomainInvariantError, match="cannot have severity 'none'"):
            BleedingEvent(severity=BleedingSeverity.NONE, start_s=1.0, end_s=2.0, confidence=0.9)

    def test_achieved_criterion_requires_a_timestamp(self):
        """Without it the gate cannot check the view preceded the division."""
        with pytest.raises(DomainInvariantError, match="carries no timestamp"):
            CvsObservation(criterion=CvsCriterion.TRIANGLE_CLEARED, achieved=True, confidence=0.9)

    def test_unknown_criterion_needs_no_timestamp(self):
        obs = CvsObservation(criterion=CvsCriterion.TRIANGLE_CLEARED, achieved=None)
        assert obs.achieved is None

    def test_not_achieved_criterion_needs_no_timestamp(self):
        obs = CvsObservation(
            criterion=CvsCriterion.TRIANGLE_CLEARED, achieved=False, confidence=0.8
        )
        assert obs.achieved is False

    def test_duplicate_criteria_are_rejected(self):
        """Two verdicts for one criterion is a contradiction, not extra evidence."""
        with pytest.raises(DomainInvariantError, match="reports a CVS criterion twice"):
            PerceptionResult(
                backend="x",
                backend_version="1",
                media_sha256=("a" * 64,),
                duration_s=100.0,
                cvs=(
                    CvsObservation(
                        criterion=CvsCriterion.TRIANGLE_CLEARED,
                        achieved=True,
                        at_s=1.0,
                        confidence=0.9,
                    ),
                    CvsObservation(
                        criterion=CvsCriterion.TRIANGLE_CLEARED, achieved=False, confidence=0.9
                    ),
                ),
            )

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_confidence_outside_zero_to_one_is_rejected(self, bad):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CvsObservation(criterion=CvsCriterion.TRIANGLE_CLEARED, achieved=None, confidence=bad)


class TestResultQueries:
    def _result(self, **kw: object) -> PerceptionResult:
        base: dict[str, object] = {
            "backend": "x",
            "backend_version": "1",
            "media_sha256": ("a" * 64,),
            "duration_s": 100.0,
        }
        return PerceptionResult(**(base | kw))

    def test_phase_span_merges_repeated_phases(self):
        """Phases recur; the span is the outer envelope."""
        out = self._result(
            phases=(
                PhaseSegment(
                    phase=SurgicalPhase.CLIPPING_AND_CUTTING,
                    start_s=60.0,
                    end_s=70.0,
                    confidence=0.9,
                ),
                PhaseSegment(
                    phase=SurgicalPhase.CLIPPING_AND_CUTTING,
                    start_s=80.0,
                    end_s=90.0,
                    confidence=0.9,
                ),
            )
        )
        assert out.phase_span(SurgicalPhase.CLIPPING_AND_CUTTING) == (60.0, 90.0)

    def test_absent_phase_returns_none_not_a_default_span(self):
        """A gate must be able to tell 'never happened' from 'happened at 0'."""
        assert self._result().phase_span(SurgicalPhase.CLIPPING_AND_CUTTING) is None

    def test_worst_bleeding_with_no_events_is_none(self):
        assert self._result().worst_bleeding() is BleedingSeverity.NONE

    def test_worst_bleeding_uses_explicit_rank_not_declaration_order(self):
        out = self._result(
            bleeding_events=(
                BleedingEvent(
                    severity=BleedingSeverity.MAJOR, start_s=1.0, end_s=2.0, confidence=0.9
                ),
                BleedingEvent(
                    severity=BleedingSeverity.MINOR, start_s=3.0, end_s=4.0, confidence=0.9
                ),
            )
        )
        assert out.worst_bleeding() is BleedingSeverity.MAJOR

    def test_cvs_observation_lookup(self):
        obs = CvsObservation(
            criterion=CvsCriterion.TWO_STRUCTURES_ONLY, achieved=None, confidence=0.0
        )
        out = self._result(cvs=(obs,))
        assert out.cvs_observation(CvsCriterion.TWO_STRUCTURES_ONLY) is obs
        assert out.cvs_observation(CvsCriterion.TRIANGLE_CLEARED) is None
