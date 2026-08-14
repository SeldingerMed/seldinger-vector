"""Episode ingestion.

Turns a customer-supplied manifest into domain objects. Two things happen
here that are worth naming:

* **Identifiers are minted, never accepted.** A manifest carries the
  customer's own opaque references; OR-Audit assigns its own ids. Accepting
  caller-chosen ids would let a customer-side identifier -- plausibly an MRN
  or a name -- become the primary key of a record we export.
* **Media arrives RAW.** Ingestion never produces cleared media. Everything it
  builds must pass through :mod:`or_audit.deid` before anything downstream can
  read it (PLAN.md section 8).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from or_audit.audit.trail import Actor, AuditAction, AuditTrail
from or_audit.domain.entities import Episode, ExternalRef, MediaAsset
from or_audit.domain.enums import DeidStatus, MediaKind, SkillBand
from or_audit.domain.ids import (
    InstitutionId,
    ProcedureId,
    RoboticSystemId,
    SurgeonId,
    new_episode_id,
    new_media_asset_id,
)
from or_audit.errors import DomainInvariantError
from or_audit.primitives import Sha256Hex


class MediaManifest(BaseModel):
    """One file offered for ingestion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: MediaKind
    uri: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    sha256: Sha256Hex
    duration_seconds: Annotated[float, Field(gt=0)] | None = None
    frame_rate: Annotated[float, Field(gt=0)] | None = None
    #: Wall-clock instant of this stream's first sample. Used to align
    #: kinematics against video; optional, because alignment is only possible
    #: when the customer's capture systems agree on a clock.
    starts_at: datetime | None = None

    @model_validator(mode="after")
    def _check_clock(self) -> Self:
        if self.starts_at is not None and self.starts_at.tzinfo is None:
            msg = "media starts_at must be timezone-aware"
            raise DomainInvariantError(msg)
        return self


class EpisodeManifest(BaseModel):
    """A case offered for ingestion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    institution_id: InstitutionId
    procedure_id: ProcedureId
    surgeon_id: SurgeonId
    system_id: RoboticSystemId
    band_at_episode: SkillBand
    performed_at: datetime
    #: The customer's own case reference, echoed back on reports. Never used
    #: as an identifier internally.
    external_ref: ExternalRef
    media: tuple[MediaManifest, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.performed_at.tzinfo is None:
            msg = "manifest performed_at must be timezone-aware"
            raise DomainInvariantError(msg)
        if not any(m.kind is MediaKind.ENDOSCOPIC_VIDEO for m in self.media):
            msg = (
                "manifest has no endoscopic video; video is the required "
                "common denominator (PLAN.md section 7)"
            )
            raise DomainInvariantError(msg)
        digests = [m.sha256 for m in self.media]
        if len(set(digests)) != len(digests):
            msg = "manifest lists the same file digest more than once"
            raise DomainInvariantError(msg)
        return self


def ingest_episode(
    manifest: EpisodeManifest,
    *,
    trail: AuditTrail | None = None,
    actor: Actor | None = None,
) -> Episode:
    """Build an episode from a manifest.

    Args:
        manifest: The case to ingest.
        trail: Optional audit trail.
        actor: Required when ``trail`` is supplied.

    Returns:
        An episode whose media are all ``RAW``.
    """
    episode_id = new_episode_id()
    media = tuple(
        MediaAsset(
            id=new_media_asset_id(),
            episode_id=episode_id,
            kind=entry.kind,
            raw_uri=entry.uri,
            sha256=entry.sha256,
            duration_seconds=entry.duration_seconds,
            frame_rate=entry.frame_rate,
            deid_status=DeidStatus.RAW,
        )
        for entry in manifest.media
    )
    episode = Episode(
        id=episode_id,
        institution_id=manifest.institution_id,
        procedure_id=manifest.procedure_id,
        surgeon_id=manifest.surgeon_id,
        system_id=manifest.system_id,
        band_at_episode=manifest.band_at_episode,
        performed_at=manifest.performed_at,
        media=media,
    )

    if trail is not None:
        if actor is None:
            msg = "an audit trail was supplied without an actor; every entry needs a principal"
            raise ValueError(msg)
        trail.append(
            actor=actor,
            action=AuditAction.EPISODE_REGISTERED,
            subject_ref=episode.id,
            payload={
                "media_count": len(media),
                "kinds": sorted({m.kind.value for m in media}),
                "has_kinematics": episode.has_kinematics,
            },
        )
        for asset in media:
            trail.append(
                actor=actor,
                action=AuditAction.MEDIA_REGISTERED,
                subject_ref=asset.id,
                payload={"kind": asset.kind.value, "sha256": asset.sha256},
            )
    return episode
