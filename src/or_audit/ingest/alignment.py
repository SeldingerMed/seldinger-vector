"""Aligning kinematics against video.

Kinematics is optional enrichment (PLAN.md section 7, V-1), so every function
here tolerates its absence and none of them is on a required path. What they
must not do is align badly and pretend otherwise: a silently wrong offset
would attribute an instrument motion to the wrong moment of the case, which is
worse than having no kinematics at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from or_audit.errors import DomainInvariantError


@dataclass(frozen=True)
class StreamWindow:
    """A stream's wall-clock extent."""

    starts_at: datetime
    duration_seconds: float

    def __post_init__(self) -> None:
        if self.starts_at.tzinfo is None:
            msg = "stream starts_at must be timezone-aware"
            raise DomainInvariantError(msg)
        if self.duration_seconds <= 0:
            msg = f"stream duration must be positive, got {self.duration_seconds}"
            raise DomainInvariantError(msg)

    @property
    def ends_at_offset_s(self) -> float:
        """Duration, named for use in overlap arithmetic."""
        return self.duration_seconds


@dataclass(frozen=True)
class Alignment:
    """How a secondary stream maps onto the reference stream's timeline."""

    #: Seconds to add to a reference-stream timestamp to get the secondary
    #: stream's own timestamp. Negative when the secondary started first.
    offset_seconds: float
    #: Span, in reference-stream time, where both streams have data.
    overlap_start_s: float
    overlap_end_s: float

    @property
    def overlap_seconds(self) -> float:
        """Length of the usable overlap."""
        return self.overlap_end_s - self.overlap_start_s


def align(
    reference: StreamWindow,
    secondary: StreamWindow,
    *,
    min_overlap_seconds: float = 1.0,
) -> Alignment:
    """Align ``secondary`` onto ``reference``'s timeline.

    Args:
        reference: Usually the endoscopic video.
        secondary: Usually the kinematics stream.
        min_overlap_seconds: Reject alignments yielding less overlap than
            this. A near-zero overlap almost always means the two clocks
            disagree, not that the case really was that short.

    Returns:
        The offset and the overlapping span in reference time.

    Raises:
        DomainInvariantError: If the streams do not overlap enough. Refusing
            is deliberate: a caller that gets an Alignment can trust it.
    """
    offset_seconds = (secondary.starts_at - reference.starts_at).total_seconds()
    overlap_start_s = max(0.0, offset_seconds)
    overlap_end_s = min(reference.duration_seconds, offset_seconds + secondary.duration_seconds)
    overlap = overlap_end_s - overlap_start_s
    if overlap < min_overlap_seconds:
        msg = (
            f"streams overlap by {overlap:.3f}s, below the {min_overlap_seconds}s "
            f"minimum; the capture clocks are probably not synchronized"
        )
        raise DomainInvariantError(msg)
    return Alignment(
        offset_seconds=offset_seconds,
        overlap_start_s=overlap_start_s,
        overlap_end_s=overlap_end_s,
    )


def try_align(
    reference: StreamWindow | None,
    secondary: StreamWindow | None,
    *,
    min_overlap_seconds: float = 1.0,
) -> Alignment | None:
    """Align when both streams are present and timed, otherwise return ``None``.

    The tolerant entry point. Absent kinematics, or a capture system that does
    not stamp wall-clock starts, is a normal condition and must not raise --
    it just means the episode is scored on video alone.
    """
    if reference is None or secondary is None:
        return None
    return align(reference, secondary, min_overlap_seconds=min_overlap_seconds)
