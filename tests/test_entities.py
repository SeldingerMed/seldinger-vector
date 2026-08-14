"""Domain invariants from PLAN.md sections 7 and 8."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from or_audit.domain.entities import Episode, MediaAsset
from or_audit.domain.enums import DeidStatus, MediaKind
from or_audit.domain.ids import (
    new_episode_id,
    new_institution_id,
    new_media_asset_id,
    new_procedure_id,
    new_surgeon_id,
    new_system_id,
)
from or_audit.errors import DeidentificationBoundaryError, DomainInvariantError

from .conftest import make_media, sha


def _episode_kwargs(episode_id: str) -> dict[str, object]:
    """Minimal valid kwargs for a directly constructed Episode."""
    return {
        "id": episode_id,
        "institution_id": new_institution_id(),
        "procedure_id": new_procedure_id(),
        "surgeon_id": new_surgeon_id(),
        "system_id": new_system_id(),
        "band_at_episode": "attending",
        "performed_at": datetime.fromisoformat("2026-03-04T14:30:00+00:00"),
    }


class TestVideoIsRequiredKinematicsIsNot:
    """Section 7: video is the common denominator; kinematics never blocks."""

    def test_episode_without_endoscopic_video_is_rejected(self, make_episode):
        with pytest.raises(DomainInvariantError, match="no endoscopic video"):
            make_episode(media_for=lambda eid: (make_media(eid, MediaKind.KINEMATICS),))

    def test_kinematics_only_episode_is_rejected_even_though_richer(self, make_episode):
        """Richer signal does not substitute for the required common denominator."""
        with pytest.raises(DomainInvariantError):
            make_episode(
                media_for=lambda eid: (
                    make_media(eid, MediaKind.KINEMATICS, tag="k1"),
                    make_media(eid, MediaKind.KINEMATICS, tag="k2"),
                )
            )

    def test_video_only_episode_is_valid(self, episode):
        assert len(episode.endoscopic_video) == 1
        assert episode.has_kinematics is False
        assert episode.kinematics == ()

    def test_kinematics_is_additive(self, make_episode):
        built = make_episode(
            media_for=lambda eid: (
                make_media(eid, MediaKind.ENDOSCOPIC_VIDEO, tag="v"),
                make_media(eid, MediaKind.KINEMATICS, tag="k"),
            )
        )
        assert built.has_kinematics is True
        assert len(built.kinematics) == 1
        assert len(built.endoscopic_video) == 1


class TestMediaConsistency:
    def test_media_from_another_episode_is_rejected(self, make_episode):
        with pytest.raises(DomainInvariantError, match="another episode"):
            make_episode(media_for=lambda _eid: (make_media(new_episode_id()),))

    def test_duplicate_media_ids_rejected(self):
        episode_id = new_episode_id()
        asset = make_media(episode_id)
        with pytest.raises(DomainInvariantError, match="duplicate media"):
            Episode(**_episode_kwargs(episode_id), media=(asset, asset))

    def test_naive_performed_at_rejected(self):
        episode_id = new_episode_id()
        kwargs = _episode_kwargs(episode_id)
        kwargs["performed_at"] = datetime(2026, 3, 4, 14, 30)
        with pytest.raises(DomainInvariantError, match="timezone-aware"):
            Episode(**kwargs, media=(make_media(episode_id),))


class TestDeidentificationGate:
    """Section 8: de-identification is a gate, not an advisory flag."""

    def test_attested_media_is_readable(self, episode):
        episode.require_readable()

    @pytest.mark.parametrize("status", [DeidStatus.RAW, DeidStatus.IN_PROGRESS, DeidStatus.FAILED])
    def test_unattested_media_cannot_be_read(self, make_episode, status):
        built = make_episode(media_for=lambda eid: (make_media(eid, deid=status),))
        with pytest.raises(DeidentificationBoundaryError, match="only 'attested'"):
            built.require_readable()

    def test_partial_attestation_blocks_the_whole_episode(self, make_episode):
        """One raw asset must block the episode, not be silently skipped."""
        built = make_episode(
            media_for=lambda eid: (
                make_media(eid, MediaKind.ENDOSCOPIC_VIDEO, tag="v"),
                make_media(eid, MediaKind.AUDIO, deid=DeidStatus.RAW, tag="a"),
            )
        )
        with pytest.raises(DeidentificationBoundaryError):
            built.require_readable()

    def test_attested_status_requires_attestation_digest(self):
        with pytest.raises(DomainInvariantError, match="no attestation digest"):
            MediaAsset(
                id=new_media_asset_id(),
                episode_id=new_episode_id(),
                kind=MediaKind.ENDOSCOPIC_VIDEO,
                uri="s3://x/a.mp4",
                sha256=sha("a"),
                deid_status=DeidStatus.ATTESTED,
            )

    def test_unattested_status_forbids_attestation_digest(self):
        with pytest.raises(DomainInvariantError, match="status is raw"):
            MediaAsset(
                id=new_media_asset_id(),
                episode_id=new_episode_id(),
                kind=MediaKind.ENDOSCOPIC_VIDEO,
                uri="s3://x/a.mp4",
                sha256=sha("a"),
                deid_status=DeidStatus.RAW,
                deid_attestation_sha256=sha("att"),
            )


class TestAggregateDeidStatus:
    @pytest.mark.parametrize(
        ("statuses", "expected"),
        [
            ((DeidStatus.ATTESTED,), DeidStatus.ATTESTED),
            ((DeidStatus.ATTESTED, DeidStatus.ATTESTED), DeidStatus.ATTESTED),
            ((DeidStatus.RAW,), DeidStatus.RAW),
            ((DeidStatus.RAW, DeidStatus.RAW), DeidStatus.RAW),
            ((DeidStatus.ATTESTED, DeidStatus.RAW), DeidStatus.IN_PROGRESS),
            ((DeidStatus.IN_PROGRESS, DeidStatus.RAW), DeidStatus.IN_PROGRESS),
            ((DeidStatus.ATTESTED, DeidStatus.FAILED), DeidStatus.FAILED),
            ((DeidStatus.RAW, DeidStatus.FAILED), DeidStatus.FAILED),
        ],
    )
    def test_aggregate_status(self, make_episode, statuses, expected):
        built = make_episode(
            media_for=lambda eid: tuple(
                make_media(eid, MediaKind.ENDOSCOPIC_VIDEO, deid=s, tag=f"t{i}")
                for i, s in enumerate(statuses)
            )
        )
        assert built.deid_status is expected

    def test_failure_dominates_regardless_of_position(self, make_episode):
        """A failed asset must not be masked by attested siblings ordered after it."""
        built = make_episode(
            media_for=lambda eid: (
                make_media(eid, MediaKind.ENDOSCOPIC_VIDEO, deid=DeidStatus.FAILED, tag="f"),
                make_media(eid, MediaKind.ENDOSCOPIC_VIDEO, tag="v"),
            )
        )
        assert built.deid_status is DeidStatus.FAILED


class TestImmutability:
    def test_entities_are_frozen(self, episode):
        with pytest.raises(ValidationError):
            episode.performed_at = datetime.fromisoformat("2027-01-01T00:00:00+00:00")

    def test_unknown_fields_rejected(self, institution):
        with pytest.raises(ValidationError):
            type(institution)(**institution.model_dump(), surprise=1)
