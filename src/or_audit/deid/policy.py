"""De-identification policy.

Policy is data, versioned and carried on the attestation, because "what rules
were in force when this file was cleared" is the first question asked when a
release is challenged (PLAN.md section 9).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class AudioDisposition(StrEnum):
    """What to do with intraoperative audio."""

    #: Destroy it. PLAN.md section 8 makes this the default: OR audio carries
    #: names, identifiers, and clinically sensitive discussion, and no part of
    #: the wedge scoring model consumes it.
    DISCARD = "discard"
    #: Keep it, subject to a documented review. Requires a justification, so
    #: that departing from the default is a deliberate, recorded act.
    RETAIN_WITH_REVIEW = "retain_with_review"


class DeidPolicy(BaseModel):
    """Rules governing a de-identification run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Bumped whenever any default below changes, so an attestation written
    #: under old rules is not mistaken for one written under current rules.
    version: str = "1"

    audio: AudioDisposition = AudioDisposition.DISCARD
    #: Why audio is being retained. Required when departing from the default.
    audio_retention_justification: str | None = None

    #: Room-facing video is discarded by default for the same reason as audio:
    #: it is almost entirely faces, and nothing downstream consumes it.
    discard_room_video: bool = True

    redact_out_of_body: bool = True
    redact_overlays: bool = True

    out_of_body_threshold: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.40
    out_of_body_min_duration_s: Annotated[float, Field(ge=0.0)] = 0.5
    analysis_stride_frames: Annotated[int, Field(ge=1)] = 15

    overlay_stride_frames: Annotated[int, Field(ge=1)] = 30
    overlay_max_std: Annotated[float, Field(ge=0.0)] = 2.0
    overlay_block_px: Annotated[int, Field(ge=1)] = 16

    def model_post_init(self, _context: object, /) -> None:
        """Require a justification when audio is retained."""
        if (
            self.audio is AudioDisposition.RETAIN_WITH_REVIEW
            and not self.audio_retention_justification
        ):
            msg = (
                "retaining intraoperative audio departs from the section 8 "
                "default and requires audio_retention_justification"
            )
            raise ValueError(msg)
