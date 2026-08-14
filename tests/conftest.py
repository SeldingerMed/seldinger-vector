"""Shared factories.

Factories default to a valid, minimal, scoreable episode so each test only
states the thing it is actually about.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from or_audit.audit.trail import Actor, ActorKind
from or_audit.domain.entities import (
    Episode,
    Institution,
    MediaAsset,
    Procedure,
    RoboticSystem,
    Surgeon,
)
from or_audit.domain.enums import (
    DeidStatus,
    Jurisdiction,
    MediaKind,
    RobotPlatform,
    SkillBand,
)
from or_audit.domain.ids import (
    new_episode_id,
    new_institution_id,
    new_media_asset_id,
    new_procedure_id,
    new_surgeon_id,
    new_system_id,
)


def sha(text: str) -> str:
    """Deterministic digest helper for fixture data."""
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def frozen_clock() -> Callable[[], datetime]:
    """A clock advancing one second per call, so chains are stable."""
    counter = itertools.count()

    def clock() -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC).replace(second=next(counter) % 60)

    return clock


@pytest.fixture
def actor() -> Actor:
    return Actor(kind=ActorKind.SERVICE, ref="test-harness")


@pytest.fixture
def institution() -> Institution:
    return Institution(
        id=new_institution_id(),
        display_name="Test Health System",
        jurisdiction=Jurisdiction.US_STATE,
    )


@pytest.fixture
def procedure() -> Procedure:
    return Procedure(
        id=new_procedure_id(),
        code="CHOLE-ROB",
        display_name="Robotic cholecystectomy",
        cvs_applicable=True,
    )


@pytest.fixture
def surgeon(institution: Institution) -> Surgeon:
    return Surgeon(
        id=new_surgeon_id(),
        institution_id=institution.id,
        external_ref="attending-0041",
        band=SkillBand.ATTENDING,
    )


@pytest.fixture
def system(institution: Institution) -> RoboticSystem:
    return RoboticSystem(
        id=new_system_id(),
        institution_id=institution.id,
        platform=RobotPlatform.HUGO,
        model_label="Hugo RAS",
    )


def make_media(
    episode_id: str,
    kind: MediaKind = MediaKind.ENDOSCOPIC_VIDEO,
    *,
    deid: DeidStatus = DeidStatus.ATTESTED,
    tag: str = "a",
) -> MediaAsset:
    """Build a media asset, attested by default."""
    return MediaAsset(
        id=new_media_asset_id(),
        episode_id=episode_id,
        kind=kind,
        uri=f"s3://deid/{tag}.mp4",
        sha256=sha(f"{episode_id}:{tag}"),
        duration_seconds=1800.0,
        frame_rate=30.0,
        deid_status=deid,
        deid_attestation_sha256=(sha(f"att:{tag}") if deid is DeidStatus.ATTESTED else None),
    )


@pytest.fixture
def make_episode(
    institution: Institution,
    procedure: Procedure,
    surgeon: Surgeon,
    system: RoboticSystem,
) -> Callable[..., Episode]:
    """Factory producing a valid episode.

    ``media_for`` receives the freshly minted episode id so callers can build
    media that belongs to the episode under construction.
    """

    def build(
        *,
        media_for: Callable[[str], tuple[MediaAsset, ...]] | None = None,
        band: SkillBand | None = None,
    ) -> Episode:
        episode_id = new_episode_id()
        media = media_for(episode_id) if media_for else (make_media(episode_id),)
        return Episode(
            id=episode_id,
            institution_id=institution.id,
            procedure_id=procedure.id,
            surgeon_id=surgeon.id,
            system_id=system.id,
            band_at_episode=band or surgeon.band,
            performed_at=datetime(2026, 3, 4, 14, 30, tzinfo=UTC),
            media=media,
        )

    return build


@pytest.fixture
def episode(make_episode: Callable[..., Episode]) -> Episode:
    return make_episode()
