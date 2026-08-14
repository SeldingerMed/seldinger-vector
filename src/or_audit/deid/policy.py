"""De-identification policy.

Policy is data, versioned and carried on the attestation, because "what rules
were in force when this file was cleared" is the first question asked when a
release is challenged (PLAN.md section 9).
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from or_audit.errors import DeidentificationBoundaryError

#: Overlay thickness at or below which a configuration is treated as capable of
#: attesting, in pixels.
#:
#: NOT a validated figure. It is a conservative engineering default chosen
#: because it is fine enough that no plausible burned-in identifier is thinner,
#: and it has not been measured against real capture systems. Establishing the
#: actual minimum rendered text size across the capture hardware in scope is an
#: open question of the same kind as PLAN.md section V's items, and until it is
#: answered this constant is an assumption, not a finding.
#:
#: Being an assumption is why it gates attestation rather than merely annotating
#: it: an attestation is a claim about what was removed, and a claim resting on
#: an unvalidated coverage bound is not one this platform should make.
SAFE_OVERLAY_MIN_PX = 8


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

    #: Minimum length of an out-of-body run to report. Defaults to zero: any
    #: run at all is reported.
    #:
    #: This started at 0.5s to suppress single-frame flicker, and that was
    #: wrong. The floor cannot distinguish a washed-out frame from a genuine
    #: half-second lens wipe, so it discarded real exits -- an 8-frame exit at
    #: 30fps spans 0.27s and vanished entirely. Suppressing a false positive
    #: costs a redacted frame of anatomy; suppressing a true positive puts the
    #: room in an attested recording. The floor is retained as a knob for
    #: callers who have measured their own footage and want it.
    out_of_body_min_duration_s: Annotated[float, Field(ge=0.0)] = 0.0

    #: Analyse every Nth frame. Defaults to 1, i.e. every frame.
    #:
    #: Sampling bounds detection recall: an out-of-body run shorter than the
    #: stride can fall entirely between two samples and never be seen. At the
    #: previous default of 15, a 13-frame lens-clean was invisible and its
    #: frames were written into output attested as de-identified. Raising this
    #: is a legitimate speed trade, but it is not free, so the resulting plan
    #: and attestation both record the recall bound it implies.
    analysis_stride_frames: Annotated[int, Field(ge=1)] = 1
    #: Why recall is being bounded. Required whenever the stride exceeds 1,
    #: for the same reason audio retention needs one: departing from the safe
    #: default must be a deliberate, recorded act rather than a config value
    #: nobody revisits.
    sampling_justification: str | None = None

    overlay_stride_frames: Annotated[int, Field(ge=1)] = 30
    overlay_max_std: Annotated[float, Field(ge=0.0)] = 2.0
    overlay_block_px: Annotated[int, Field(ge=1)] = 16
    #: Fraction of a block's pixels that must be static to seed it. Lower
    #: values detect thinner overlays at the cost of masking more benign
    #: static anatomy.
    overlay_min_static_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    #: Why a coarse overlay grid is being used. Required to *analyse* with one.
    #: It does not permit attestation -- see
    #: :attr:`guarantees_overlay_coverage`.
    overlay_recall_justification: str | None = None

    @property
    def overlay_min_detectable_px(self) -> int:
        """Thinnest overlay this configuration is guaranteed to detect.

        A block is seeded only when ``min_static_fraction`` of its pixels are
        static, so an overlay must span at least that fraction of a block edge
        to seed anything. Below this thickness the overlay may be missed
        entirely, at any dilation.
        """
        return max(1, math.ceil(self.overlay_block_px * self.overlay_min_static_fraction))

    @property
    def guarantees_overlay_coverage(self) -> bool:
        """Whether this configuration may be used to attest.

        False when the overlay grid is too coarse to guarantee coverage of the
        thinnest identifier the platform is willing to assume exists. Such a
        policy is still useful -- triage, archive backfill, deciding what needs
        a finer pass -- but :func:`or_audit.deid.pipeline.redact` will not
        produce an attested asset with it. Analysis and attestation are
        different claims and this is the line between them.
        """
        return self.overlay_min_detectable_px <= SAFE_OVERLAY_MIN_PX

    @model_validator(mode="after")
    def _require_justifications(self) -> Self:
        """Require justifications for departures from the safe defaults.

        Raised as ``DeidentificationBoundaryError`` rather than ``ValueError``.
        pydantic wraps ``ValueError`` from a validator into its own
        ``ValidationError``, so a caller guarding the de-identification
        boundary with ``except DeidentificationBoundaryError`` would silently
        miss a rejected policy -- the error taxonomy would leak pydantic
        internals at exactly the boundary it exists to describe. Exceptions
        that do not derive from ``ValueError`` propagate unwrapped, which is
        also how the domain entities raise ``DomainInvariantError``.
        """
        if (
            self.audio is AudioDisposition.RETAIN_WITH_REVIEW
            and not self.audio_retention_justification
        ):
            msg = (
                "retaining intraoperative audio departs from the section 8 "
                "default and requires audio_retention_justification"
            )
            raise DeidentificationBoundaryError(msg)
        if not self.guarantees_overlay_coverage and not self.overlay_recall_justification:
            msg = (
                f"an overlay grid of {self.overlay_block_px}px at a "
                f"{self.overlay_min_static_fraction:g} static fraction can only "
                f"guarantee detection of overlays at least "
                f"{self.overlay_min_detectable_px}px thick, above the "
                f"{SAFE_OVERLAY_MIN_PX}px attesting bound. Such a policy may be "
                f"used for analysis, but it cannot attest, and using it at all "
                f"requires overlay_recall_justification"
            )
            raise DeidentificationBoundaryError(msg)
        if self.analysis_stride_frames > 1 and not self.sampling_justification:
            msg = (
                f"an analysis stride of {self.analysis_stride_frames} cannot detect "
                f"out-of-body runs shorter than {self.analysis_stride_frames} frames, "
                f"so material can reach an attested recording; this is a deliberate "
                f"recall trade and requires sampling_justification"
            )
            raise DeidentificationBoundaryError(msg)
        return self
