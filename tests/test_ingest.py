"""Ingestion: manifests into episodes, and stream alignment."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from or_audit.audit.trail import Actor, ActorKind, AuditAction, AuditTrail
from or_audit.domain.enums import DeidStatus, MediaKind, SkillBand
from or_audit.domain.ids import (
    new_institution_id,
    new_procedure_id,
    new_surgeon_id,
    new_system_id,
)
from or_audit.errors import DomainInvariantError
from or_audit.ingest.alignment import StreamWindow, align, try_align
from or_audit.ingest.manifest import EpisodeManifest, MediaManifest, ingest_episode

START = datetime(2026, 3, 4, 14, 30, tzinfo=UTC)


def media(kind: MediaKind = MediaKind.ENDOSCOPIC_VIDEO, digest: str = "a") -> MediaManifest:
    return MediaManifest(
        kind=kind,
        uri=f"s3://raw/{digest}.bin",
        sha256=digest * 64,
        duration_seconds=1800.0,
        frame_rate=30.0,
    )


def manifest(*entries: MediaManifest) -> EpisodeManifest:
    return EpisodeManifest(
        institution_id=new_institution_id(),
        procedure_id=new_procedure_id(),
        surgeon_id=new_surgeon_id(),
        system_id=new_system_id(),
        band_at_episode=SkillBand.ATTENDING,
        performed_at=START,
        external_ref="CASE-2026-0041",
        media=entries or (media(),),
    )


class TestManifestValidation:
    def test_manifest_without_video_is_rejected(self):
        with pytest.raises(DomainInvariantError, match="no endoscopic video"):
            manifest(media(MediaKind.KINEMATICS, "b"))

    def test_duplicate_digests_are_rejected(self):
        """The same file listed twice is a manifest bug, not two recordings."""
        with pytest.raises(DomainInvariantError, match="same file digest"):
            manifest(media(), media(MediaKind.KINEMATICS, "a"))

    def test_naive_performed_at_is_rejected(self):
        with pytest.raises(DomainInvariantError, match="timezone-aware"):
            EpisodeManifest(
                institution_id=new_institution_id(),
                procedure_id=new_procedure_id(),
                surgeon_id=new_surgeon_id(),
                system_id=new_system_id(),
                band_at_episode=SkillBand.ATTENDING,
                performed_at=datetime(2026, 3, 4, 14, 30),
                external_ref="CASE-1",
                media=(media(),),
            )

    def test_naive_media_start_is_rejected(self):
        with pytest.raises(DomainInvariantError, match="timezone-aware"):
            MediaManifest(
                kind=MediaKind.KINEMATICS,
                uri="s3://raw/k.bin",
                sha256="c" * 64,
                starts_at=datetime(2026, 3, 4, 14, 30),
            )


class TestIngestEpisode:
    def test_media_arrives_raw(self):
        """Ingestion never produces cleared media (PLAN.md section 8)."""
        episode = ingest_episode(manifest())
        assert episode.deid_status is DeidStatus.RAW
        assert all(a.deid_status is DeidStatus.RAW for a in episode.media)

    def test_identifiers_are_minted_not_taken_from_the_manifest(self):
        """A customer reference must never become our primary key."""
        episode = ingest_episode(manifest())
        assert episode.id.startswith("epi_")
        assert all(a.id.startswith("med_") for a in episode.media)
        assert "CASE-2026-0041" not in episode.id

    def test_media_is_bound_to_the_new_episode(self):
        episode = ingest_episode(manifest())
        assert all(a.episode_id == episode.id for a in episode.media)

    def test_kinematics_is_carried_through_when_present(self):
        episode = ingest_episode(manifest(media(), media(MediaKind.KINEMATICS, "b")))
        assert episode.has_kinematics is True

    def test_video_only_episode_is_valid(self):
        assert ingest_episode(manifest()).has_kinematics is False

    def test_two_ingests_of_the_same_manifest_get_distinct_ids(self):
        """Re-ingesting must create a new episode, not silently collide."""
        assert ingest_episode(manifest()).id != ingest_episode(manifest()).id


class TestIngestAuditing:
    def test_registration_is_recorded_for_episode_and_each_asset(self):
        trail = AuditTrail()
        actor = Actor(kind=ActorKind.CUSTOMER_SYSTEM, ref="customer-import")
        episode = ingest_episode(
            manifest(media(), media(MediaKind.KINEMATICS, "b")), trail=trail, actor=actor
        )
        assert [e.action for e in trail] == [
            AuditAction.EPISODE_REGISTERED,
            AuditAction.MEDIA_REGISTERED,
            AuditAction.MEDIA_REGISTERED,
        ]
        assert trail.entries[0].subject_ref == episode.id
        trail.verify()

    def test_trail_without_actor_is_rejected(self):
        with pytest.raises(ValueError, match="without an actor"):
            ingest_episode(manifest(), trail=AuditTrail())


class TestAlignment:
    def test_simultaneous_streams_have_zero_offset(self):
        window = StreamWindow(starts_at=START, duration_seconds=100.0)
        result = align(window, StreamWindow(starts_at=START, duration_seconds=100.0))
        assert result.offset_seconds == 0.0
        assert result.overlap_seconds == 100.0

    def test_late_secondary_gives_a_positive_offset(self):
        video = StreamWindow(starts_at=START, duration_seconds=100.0)
        kinematics = StreamWindow(starts_at=START + timedelta(seconds=10), duration_seconds=100.0)
        result = align(video, kinematics)
        assert result.offset_seconds == 10.0
        assert result.overlap_start_s == 10.0
        assert result.overlap_end_s == 100.0

    def test_early_secondary_gives_a_negative_offset(self):
        video = StreamWindow(starts_at=START, duration_seconds=100.0)
        kinematics = StreamWindow(starts_at=START - timedelta(seconds=10), duration_seconds=100.0)
        result = align(video, kinematics)
        assert result.offset_seconds == -10.0
        assert result.overlap_start_s == 0.0
        assert result.overlap_end_s == 90.0

    def test_disjoint_streams_are_refused(self):
        """A wrong offset misattributes motion; refusing beats guessing."""
        video = StreamWindow(starts_at=START, duration_seconds=100.0)
        kinematics = StreamWindow(starts_at=START + timedelta(seconds=500), duration_seconds=100.0)
        with pytest.raises(DomainInvariantError, match="not synchronized"):
            align(video, kinematics)

    def test_marginal_overlap_is_refused(self):
        video = StreamWindow(starts_at=START, duration_seconds=100.0)
        kinematics = StreamWindow(starts_at=START + timedelta(seconds=99.5), duration_seconds=100.0)
        with pytest.raises(DomainInvariantError, match="below the"):
            align(video, kinematics)

    def test_marginal_overlap_is_accepted_when_the_floor_is_lowered(self):
        video = StreamWindow(starts_at=START, duration_seconds=100.0)
        kinematics = StreamWindow(starts_at=START + timedelta(seconds=99.5), duration_seconds=100.0)
        assert align(video, kinematics, min_overlap_seconds=0.1).overlap_seconds == 0.5

    @pytest.mark.parametrize("bad_duration", [0.0, -1.0])
    def test_non_positive_duration_is_rejected(self, bad_duration):
        with pytest.raises(DomainInvariantError, match="duration must be positive"):
            StreamWindow(starts_at=START, duration_seconds=bad_duration)

    def test_naive_start_is_rejected(self):
        with pytest.raises(DomainInvariantError, match="timezone-aware"):
            StreamWindow(
                starts_at=datetime(2026, 3, 4, tzinfo=None),
                duration_seconds=10.0,
            )


class TestTolerantAlignment:
    """Absent kinematics is normal and must never raise (PLAN.md V-1)."""

    def test_missing_secondary_returns_none(self):
        video = StreamWindow(starts_at=START, duration_seconds=100.0)
        assert try_align(video, None) is None

    def test_missing_reference_returns_none(self):
        assert try_align(None, StreamWindow(starts_at=START, duration_seconds=10.0)) is None

    def test_both_missing_returns_none(self):
        assert try_align(None, None) is None

    def test_present_streams_still_align(self):
        window = StreamWindow(starts_at=START, duration_seconds=100.0)
        assert try_align(window, window) is not None

    def test_present_but_disjoint_streams_still_raise(self):
        """Tolerating absence is not the same as tolerating a bad alignment."""
        video = StreamWindow(starts_at=START, duration_seconds=10.0)
        kinematics = StreamWindow(starts_at=START + timedelta(hours=1), duration_seconds=10.0)
        with pytest.raises(DomainInvariantError):
            try_align(video, kinematics)
