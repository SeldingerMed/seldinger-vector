"""Core entities: institution, surgeon, system, procedure, episode, media.

Shape follows the data model in PLAN.md section 7.1:
``procedure -> surgeon -> system -> episode -> aligned video (+kinematics)
-> annotations -> score -> decision -> contestation``. This module owns the
left-hand side up to and including media; scores, decisions and contestations
arrive in later phases.

Two invariants are enforced here rather than in a service layer, because both
are architectural commitments that a caller must not be able to opt out of:

1. **Video is required, kinematics is not** (section 7). An episode without
   in-body endoscopic video cannot be scored, and no code path may become
   blocked on kinematics being present.
2. **De-identification is a gate, not a flag** (section 8). Reading media for
   perception, scoring, reporting or export requires an attested asset.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from or_audit.domain.enums import (
    DeidStatus,
    Jurisdiction,
    MediaKind,
    RobotPlatform,
    SkillBand,
)
from or_audit.domain.ids import (
    EpisodeId,
    InstitutionId,
    MediaAssetId,
    ProcedureId,
    RoboticSystemId,
    SurgeonId,
)
from or_audit.errors import DeidentificationBoundaryError, DomainInvariantError

#: Lowercase hex SHA-256 digest.
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

#: Opaque, caller-supplied reference into a customer's own system of record.
#: Deliberately not validated as a name or MRN -- OR-Audit does not store
#: patient or surgeon identifiers, only pseudonymous handles (section 8).
ExternalRef = Annotated[str, StringConstraints(min_length=1, max_length=128)]


class _Frozen(BaseModel):
    """Base for immutable domain records."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class Institution(_Frozen):
    """A hospital, health system, or training program."""

    id: InstitutionId
    display_name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    jurisdiction: Jurisdiction
    #: Set once counsel confirms a privilege posture for this institution
    #: (PLAN.md section 9, V-3). Reporting uses it to decide which artifacts
    #: may carry individual attribution.
    peer_review_protection_confirmed: bool = False


class Surgeon(_Frozen):
    """A pseudonymous surgeon record.

    OR-Audit stores no surgeon name. ``external_ref`` is the customer's own
    handle, meaningful only inside the customer's system.
    """

    id: SurgeonId
    institution_id: InstitutionId
    external_ref: ExternalRef
    band: SkillBand


class RoboticSystem(_Frozen):
    """A specific robotic platform instance."""

    id: RoboticSystemId
    institution_id: InstitutionId
    platform: RobotPlatform
    #: Vendor's model string, free text; informational only.
    model_label: Annotated[str, StringConstraints(max_length=120)] = ""
    #: True when this system has a negotiated kinematics feed. Advisory:
    #: no scoring path may require it (section 7, V-1).
    kinematics_agreement_in_place: bool = False


class Procedure(_Frozen):
    """A procedure type, e.g. robotic cholecystectomy."""

    id: ProcedureId
    code: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    display_name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    #: Whether the Critical View of Safety rubric applies to this procedure.
    #: Only cholecystectomy-family procedures should set this.
    cvs_applicable: bool = False


class MediaAsset(_Frozen):
    """One media file belonging to an episode."""

    id: MediaAssetId
    episode_id: EpisodeId
    kind: MediaKind
    #: Storage locator. Never a path containing patient identifiers.
    uri: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    sha256: Sha256Hex
    duration_seconds: Annotated[float, Field(gt=0)] | None = None
    frame_rate: Annotated[float, Field(gt=0)] | None = None
    deid_status: DeidStatus = DeidStatus.RAW
    #: Identifier of the de-identification attestation that cleared this
    #: asset. Required when and only when ``deid_status`` is ``ATTESTED``.
    deid_attestation_sha256: Sha256Hex | None = None

    @model_validator(mode="after")
    def _attestation_matches_status(self) -> Self:
        attested = self.deid_status is DeidStatus.ATTESTED
        if attested and self.deid_attestation_sha256 is None:
            msg = f"media {self.id} is ATTESTED but carries no attestation digest"
            raise DomainInvariantError(msg)
        if not attested and self.deid_attestation_sha256 is not None:
            msg = f"media {self.id} carries an attestation digest but status is {self.deid_status}"
            raise DomainInvariantError(msg)
        return self

    @property
    def is_readable(self) -> bool:
        """Whether this asset may be read by perception, scoring, or export."""
        return self.deid_status is DeidStatus.ATTESTED

    def require_readable(self) -> None:
        """Raise unless this asset has cleared de-identification.

        Raises:
            DeidentificationBoundaryError: If the asset is not attested.
        """
        if not self.is_readable:
            msg = (
                f"media {self.id} has de-identification status "
                f"{self.deid_status.value}; only 'attested' media may be read "
                f"(PLAN.md section 8)"
            )
            raise DeidentificationBoundaryError(msg)


class Episode(_Frozen):
    """One performed case: the unit of assessment.

    An episode is the join of a procedure, a surgeon, a system, and the media
    recorded for it.
    """

    id: EpisodeId
    institution_id: InstitutionId
    procedure_id: ProcedureId
    surgeon_id: SurgeonId
    system_id: RoboticSystemId
    #: Band at the time of *this* episode. Copied rather than dereferenced
    #: from the surgeon, because bands change and a score must be interpreted
    #: against the band the surgeon held when the case was performed
    #: (section 13 stratification).
    band_at_episode: SkillBand
    performed_at: datetime
    media: tuple[MediaAsset, ...]

    @model_validator(mode="after")
    def _check_media(self) -> Self:
        if any(asset.episode_id != self.id for asset in self.media):
            msg = f"episode {self.id} contains media belonging to another episode"
            raise DomainInvariantError(msg)

        ids = [asset.id for asset in self.media]
        if len(set(ids)) != len(ids):
            msg = f"episode {self.id} contains duplicate media asset identifiers"
            raise DomainInvariantError(msg)

        if not any(asset.kind is MediaKind.ENDOSCOPIC_VIDEO for asset in self.media):
            msg = (
                f"episode {self.id} has no endoscopic video; video is the "
                f"required common denominator (PLAN.md section 7)"
            )
            raise DomainInvariantError(msg)
        return self

    @model_validator(mode="after")
    def _check_timezone(self) -> Self:
        if self.performed_at.tzinfo is None:
            msg = f"episode {self.id} performed_at must be timezone-aware"
            raise DomainInvariantError(msg)
        return self

    @property
    def endoscopic_video(self) -> tuple[MediaAsset, ...]:
        """Endoscopic video assets, in declaration order."""
        return tuple(a for a in self.media if a.kind is MediaKind.ENDOSCOPIC_VIDEO)

    @property
    def kinematics(self) -> tuple[MediaAsset, ...]:
        """Kinematics assets, if any. Absence is normal, not an error."""
        return tuple(a for a in self.media if a.kind is MediaKind.KINEMATICS)

    @property
    def has_kinematics(self) -> bool:
        """Whether kinematics enrichment is available for this episode."""
        return bool(self.kinematics)

    @property
    def deid_status(self) -> DeidStatus:
        """Aggregate de-identification status across all media.

        The episode is only ``ATTESTED`` when every asset is. Any failure
        dominates; otherwise the least-advanced asset wins.
        """
        statuses = {a.deid_status for a in self.media}
        if DeidStatus.FAILED in statuses:
            return DeidStatus.FAILED
        if statuses == {DeidStatus.ATTESTED}:
            return DeidStatus.ATTESTED
        if DeidStatus.IN_PROGRESS in statuses or DeidStatus.ATTESTED in statuses:
            return DeidStatus.IN_PROGRESS
        return DeidStatus.RAW

    def require_readable(self) -> None:
        """Raise unless every media asset has cleared de-identification.

        Raises:
            DeidentificationBoundaryError: If any asset is not attested.
        """
        for asset in self.media:
            asset.require_readable()
