"""The de-identification attestation artifact.

PLAN.md section 8 calls for "a de-identification attestation artifact per
episode, versioned, auditable, and reviewable by an institution's privacy
office", and notes that this artifact is a sales asset rather than only a
compliance obligation. It is therefore built to be read by someone hostile to
it: every field answers a question a privacy officer would ask, and the digest
covers all of them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from or_audit.audit.canonical import digest
from or_audit.deid.plan import RedactionPlan
from or_audit.deid.policy import DeidPolicy
from or_audit.domain.enums import MediaKind
from or_audit.domain.ids import EpisodeId, MediaAssetId
from or_audit.errors import DomainInvariantError
from or_audit.primitives import PrincipalRef, Sha256Hex
from or_audit.version import SCHEMA_VERSION


class DeidAttestation(BaseModel):
    """A signed-off record of one media asset's de-identification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA_VERSION
    media_id: MediaAssetId
    episode_id: EpisodeId
    media_kind: MediaKind

    performed_at: datetime
    performed_by: PrincipalRef

    policy: DeidPolicy
    plan: RedactionPlan

    #: Digest of the material that went in.
    source_sha256: Sha256Hex
    #: Digest of the material that came out, computed by the pipeline from the
    #: bytes the writer produced. Absent only when the asset was destroyed.
    output_sha256: Sha256Hex | None
    output_uri: str | None
    output_frame_count: Annotated[int, Field(ge=0)] | None

    #: Set when the asset was destroyed rather than redacted, e.g. audio under
    #: the default policy.
    discarded: bool = False
    discard_reason: str | None = None

    @model_validator(mode="after")
    def _check_disposition(self) -> Self:
        if self.discarded:
            if self.output_sha256 is not None or self.output_uri is not None:
                msg = f"attestation for {self.media_id} is discarded but names an output"
                raise DomainInvariantError(msg)
            if not self.discard_reason:
                msg = f"attestation for {self.media_id} is discarded without a reason"
                raise DomainInvariantError(msg)
            return self
        if self.output_sha256 is None or self.output_uri is None:
            msg = (
                f"attestation for {self.media_id} is not discarded and must name "
                f"the redacted output it produced"
            )
            raise DomainInvariantError(msg)
        if self.discard_reason is not None:
            msg = f"attestation for {self.media_id} names a discard reason but is not discarded"
            raise DomainInvariantError(msg)
        if self.policy.version != self.plan.policy_version:
            msg = (
                f"attestation for {self.media_id} carries policy version "
                f"{self.policy.version!r} but its plan was built under "
                f"{self.plan.policy_version!r}; the record would misdescribe "
                f"which rules were in force"
            )
            raise DomainInvariantError(msg)
        if self.source_sha256 == self.output_sha256 and not self.plan.is_noop:
            msg = (
                f"attestation for {self.media_id} claims redactions were applied "
                f"but the output is byte-identical to the source"
            )
            raise DomainInvariantError(msg)
        return self

    @property
    def digest(self) -> str:
        """Content digest of this attestation.

        Recorded on the media asset, so an asset marked attested points at the
        exact record justifying it. Changing any field changes the digest and
        breaks that link, which is the intent.
        """
        return digest(self.model_dump(mode="python"))

    def summary(self) -> dict[str, object]:
        """Compact form for audit payloads and review listings."""
        return {
            "media_id": self.media_id,
            "media_kind": self.media_kind.value,
            "discarded": self.discarded,
            "dropped_segments": len(self.plan.dropped_segments),
            "masked_boxes": len(self.plan.masked_boxes),
            "output_frame_count": self.output_frame_count,
            "min_detectable_event_seconds": self.plan.min_detectable_event_seconds,
            "recall_bounded": self.plan.is_recall_bounded,
            "attestation_sha256": self.digest,
        }
