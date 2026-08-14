"""Redaction plans and their application.

A plan is what the detectors found and what will be done about it. Keeping it
as a separate, inspectable artifact matters for two reasons: a reviewer can
see the proposed redactions before any bytes are written, and the attestation
can record exactly what was applied rather than a summary of it.

Applying a plan is real work on real pixels, not a description of work. That
distinction is the point of PLAN.md section 8: a status field that says
"clean" without bytes having changed is worse than no status at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from or_audit.deid.detectors import PixelBox, TimeSegment
from or_audit.media.frames import Frame, FrameSource


class PlannedSegment(BaseModel):
    """A time span to be dropped."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_s: Annotated[float, Field(ge=0.0)]
    end_s: Annotated[float, Field(gt=0.0)]
    reason: str


class PlannedBox(BaseModel):
    """A pixel region to be masked in every retained frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    left: Annotated[int, Field(ge=0)]
    top: Annotated[int, Field(ge=0)]
    right: Annotated[int, Field(gt=0)]
    bottom: Annotated[int, Field(gt=0)]
    reason: str


class RedactionPlan(BaseModel):
    """Everything that will be changed, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_version: str
    detectors: tuple[str, ...]
    source_frame_count: Annotated[int, Field(ge=0)]
    source_frame_rate: Annotated[float, Field(gt=0.0)]
    #: Frame stride the detectors ran at. Recorded because it bounds what the
    #: plan could possibly have found, and a reader of the attestation is
    #: entitled to know that bound rather than infer completeness.
    analysis_stride_frames: Annotated[int, Field(ge=1)] = 1
    dropped_segments: tuple[PlannedSegment, ...] = ()
    masked_boxes: tuple[PlannedBox, ...] = ()

    @property
    def min_detectable_event_seconds(self) -> float:
        """Shortest out-of-body run this analysis was guaranteed to see.

        A run spanning ``k`` consecutive frames contains at least one sampled
        frame only when ``k >= stride``. Anything shorter may have fallen
        between samples, so this is a hard bound on recall, not an estimate.
        """
        return self.analysis_stride_frames / self.source_frame_rate

    @property
    def is_recall_bounded(self) -> bool:
        """Whether sampling could have missed short events entirely."""
        return self.analysis_stride_frames > 1

    @property
    def drops_everything(self) -> bool:
        """Whether the plan would remove every frame."""
        if self.source_frame_count == 0:
            return True
        total_s = self.source_frame_count / self.source_frame_rate
        return any(s.start_s <= 0.0 and s.end_s >= total_s - 1e-9 for s in self.dropped_segments)

    @property
    def is_noop(self) -> bool:
        """Whether the plan would change nothing.

        A no-op plan is a legitimate outcome -- some recordings are clean --
        but it is worth surfacing, because it is also what a misconfigured
        detector produces.
        """
        return not self.dropped_segments and not self.masked_boxes

    def as_segments(self) -> tuple[TimeSegment, ...]:
        """Dropped spans as detector-level segments."""
        return tuple(TimeSegment(start_s=s.start_s, end_s=s.end_s) for s in self.dropped_segments)

    def as_boxes(self) -> tuple[PixelBox, ...]:
        """Masked regions as detector-level boxes."""
        return tuple(
            PixelBox(left=b.left, top=b.top, right=b.right, bottom=b.bottom)
            for b in self.masked_boxes
        )


def apply_plan(source: FrameSource, plan: RedactionPlan) -> Iterator[Frame]:
    """Apply ``plan`` to ``source``, yielding retained, masked frames.

    Dropped frames are not yielded at all rather than blanked, so the output
    contains no trace of the out-of-body material -- a blanked frame still
    tells a viewer when the camera left the body, and a blanking bug leaves
    the pixels in place.

    Timestamps are re-based onto a contiguous timeline, because the output is
    a new recording, and gaps would make every downstream duration wrong.

    Args:
        source: Frames to redact.
        plan: What to drop and mask.

    Yields:
        Retained frames with masked regions zeroed and re-based timestamps.
    """
    segments = plan.as_segments()
    boxes = plan.as_boxes()
    kept = 0
    for frame in source.iter_frames():
        if any(segment.contains(frame.timestamp_s) for segment in segments):
            continue
        pixels = frame.pixels
        if boxes:
            pixels = pixels.copy()
            for box in boxes:
                pixels[box.top : box.bottom, box.left : box.right] = 0
        yield Frame(
            index=kept,
            timestamp_s=kept / plan.source_frame_rate,
            pixels=np.ascontiguousarray(pixels),
        )
        kept += 1
