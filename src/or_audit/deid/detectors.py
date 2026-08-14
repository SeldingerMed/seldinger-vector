"""Detectors for the two PHI risks specific to endoscopic recordings.

PLAN.md section 8 names both:

* **Out-of-body segments.** The camera leaves the patient mid-procedure --
  for lens cleaning, instrument changes, at the end of the case -- and records
  the room, staff faces, and whiteboards. These are the highest-risk frames in
  the file and they are not at predictable timestamps.
* **Burned-in overlays.** Capture systems render patient name, MRN, DOB and
  date into the video raster. They cannot be stripped as metadata because they
  are pixels.

Honest scope, because PLAN.md section 9 means these claims get tested: both
detectors are **screening heuristics**, not validated classifiers. They are
deterministic, explainable, and cheap, which makes them suitable for flagging
material for redaction and for review. They are not suitable as the sole
control on a release path, and nothing here should be read as a validation
claim. Each carries a version so an attestation records exactly what ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from or_audit.media.frames import Frame, FrameSource, RgbArray, sample_indices

OUT_OF_BODY_DETECTOR: Final = "redness-ratio"
OUT_OF_BODY_DETECTOR_VERSION: Final = "1"

OVERLAY_DETECTOR: Final = "temporal-invariance"
OVERLAY_DETECTOR_VERSION: Final = "1"


@dataclass(frozen=True)
class TimeSegment:
    """A half-open time span ``[start_s, end_s)``."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.end_s <= self.start_s:
            msg = f"segment must have positive duration, got [{self.start_s}, {self.end_s})"
            raise ValueError(msg)

    @property
    def duration_s(self) -> float:
        """Length of the segment in seconds."""
        return self.end_s - self.start_s

    def contains(self, timestamp_s: float) -> bool:
        """Whether ``timestamp_s`` falls inside the segment."""
        return self.start_s <= timestamp_s < self.end_s


@dataclass(frozen=True)
class PixelBox:
    """An axis-aligned pixel region, ``[left, right)`` by ``[top, bottom)``."""

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            msg = (
                f"box must have positive area, got "
                f"({self.left}, {self.top}) to ({self.right}, {self.bottom})"
            )
            raise ValueError(msg)
        if self.left < 0 or self.top < 0:
            msg = f"box origin must be non-negative, got ({self.left}, {self.top})"
            raise ValueError(msg)

    @property
    def area(self) -> int:
        """Number of pixels covered."""
        return (self.right - self.left) * (self.bottom - self.top)


def redness_ratio(pixels: RgbArray) -> float:
    """Fraction of total channel energy carried by the red channel.

    In-body endoscopic imagery is dominated by blood and mucosa, so red
    carries roughly 0.45-0.60 of the total. Room imagery is broad-spectrum and
    sits near the achromatic 0.333. The gap is wide, which is why this crude
    statistic separates the two classes reliably enough to screen on.

    Returns:
        A value in ``[0, 1]``. Returns ``1/3`` for a pure-black frame, where
        the ratio is undefined and treating it as achromatic is the
        conservative choice -- black frames get flagged for review rather than
        assumed to be in-body.
    """
    channel_means = pixels.reshape(-1, 3).mean(axis=0)
    total = float(channel_means.sum())
    if total <= 0.0:
        return 1.0 / 3.0
    return float(channel_means[0]) / total


def detect_out_of_body(
    source: FrameSource,
    *,
    threshold: float = 0.40,
    stride: int = 15,
    min_duration_s: float = 0.5,
) -> tuple[TimeSegment, ...]:
    """Find spans where the camera appears to be outside the patient.

    Args:
        source: Frames to analyse.
        threshold: Redness ratio at or below which a frame is judged
            out-of-body. The default sits between the achromatic 0.333 and the
            in-body floor of roughly 0.45.
        stride: Analyse every ``stride``-th frame.
        min_duration_s: Discard runs shorter than this. A single dark or
            washed-out frame mid-procedure is not the camera leaving the body,
            and without this the output is unusable flicker.

    Returns:
        Merged, ascending, non-overlapping segments.
    """
    if not 0.0 < threshold < 1.0:
        msg = f"threshold must be in (0, 1), got {threshold}"
        raise ValueError(msg)
    if min_duration_s < 0:
        msg = f"min_duration_s must be non-negative, got {min_duration_s}"
        raise ValueError(msg)

    indices = sample_indices(source.frame_count, stride=stride)
    if not indices:
        return ()

    frame_period = 1.0 / source.frame_rate
    recording_end_s = source.frame_count * frame_period

    # A flagged sample covers the unsampled gap on BOTH sides of it.
    #
    # One-sided coverage leaks. At stride 15, a camera exit beginning at frame
    # 8 is first sampled at frame 15, so frames 8-14 are out-of-body material
    # that never gets dropped. That is a PHI leak, not a rounding artifact.
    #
    # The transition happened somewhere inside the gap and sampling cannot say
    # where, so both neighbouring gaps are treated as suspect. This over-drops
    # in-body frames adjacent to an exit, which is the correct direction to be
    # wrong in: losing a second of anatomy costs a second of anatomy, keeping
    # one frame of the room costs a breach.
    flagged: list[tuple[float, float]] = []
    for position, index in enumerate(indices):
        frame = source.read(index)
        if redness_ratio(frame.pixels) > threshold:
            continue
        start_s = (
            source.read(indices[position - 1]).timestamp_s if position > 0 else frame.timestamp_s
        )
        end_s = (
            source.read(indices[position + 1]).timestamp_s + frame_period
            if position + 1 < len(indices)
            else recording_end_s
        )
        flagged.append((start_s, min(end_s, recording_end_s)))

    return _merge_and_filter(
        flagged, min_duration_s=min_duration_s, recording_end_s=recording_end_s
    )


def _merge_and_filter(
    spans: list[tuple[float, float]],
    *,
    min_duration_s: float,
    recording_end_s: float,
) -> tuple[TimeSegment, ...]:
    """Merge touching spans, then drop those below the duration floor.

    Segments reaching the end of the recording are exempt from the floor. The
    floor exists to suppress single-frame flicker mid-procedure, but the
    end-of-case camera withdrawal is both the most predictable out-of-body
    event and often shorter than the floor -- five frames at 30fps is 0.17s.
    Suppressing it would drop exactly the material PLAN.md section 8 names
    first, so the floor is not allowed to reach it.
    """
    if not spans:
        return ()
    merged: list[list[float]] = [list(spans[0])]
    for start_s, end_s in spans[1:]:
        # Sampled spans abut exactly, so compare with a tolerance rather than
        # equality: float division of frame indices does not land cleanly.
        if start_s <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end_s)
        else:
            merged.append([start_s, end_s])
    return tuple(
        TimeSegment(start_s=start_s, end_s=end_s)
        for start_s, end_s in merged
        if end_s - start_s >= min_duration_s or end_s >= recording_end_s - 1e-9
    )


def detect_static_overlays(
    source: FrameSource,
    *,
    stride: int = 30,
    max_std: float = 2.0,
    block: int = 16,
    min_blocks: int = 1,
) -> tuple[PixelBox, ...]:
    """Find regions that do not change over time.

    Burned-in identifiers are rendered once and never move, so temporal
    variance separates them from anatomy, which is always moving. The frame is
    scored per pixel, reduced to a coarse block grid, and contiguous static
    blocks are merged into boxes.

    A caveat worth stating rather than burying: a genuinely motionless region
    of anatomy -- a static retractor, a letterboxed border -- also reads as
    static. Over-inclusion is the intended failure direction. Redacting a few
    extra blocks costs pixels; missing an MRN costs a breach.

    Args:
        source: Frames to analyse.
        stride: Analyse every ``stride``-th frame.
        max_std: Per-pixel standard deviation at or below which a pixel counts
            as static. Small but non-zero, to tolerate encoder noise.
        block: Grid size in pixels for merging.
        min_blocks: Discard components smaller than this many blocks.

    Returns:
        Bounding boxes in ascending ``(top, left)`` order.
    """
    if block < 1:
        msg = f"block must be at least 1, got {block}"
        raise ValueError(msg)
    if max_std < 0:
        msg = f"max_std must be non-negative, got {max_std}"
        raise ValueError(msg)

    indices = sample_indices(source.frame_count, stride=stride)
    # Variance over a single sample is zero everywhere, which would mark the
    # whole frame static. Refuse rather than emit a useless full-frame box.
    if len(indices) < 2:
        return ()

    stack = np.stack([source.read(index).pixels.astype(np.float32) for index in indices])
    static_mask = stack.std(axis=0).max(axis=2) <= max_std

    height, width = static_mask.shape
    rows = (height + block - 1) // block
    cols = (width + block - 1) // block
    grid = np.zeros((rows, cols), dtype=bool)
    for row in range(rows):
        for col in range(cols):
            patch = static_mask[row * block : (row + 1) * block, col * block : (col + 1) * block]
            grid[row, col] = bool(patch.all())

    boxes = [
        PixelBox(
            left=min(c for _, c in component) * block,
            top=min(r for r, _ in component) * block,
            right=min((max(c for _, c in component) + 1) * block, width),
            bottom=min((max(r for r, _ in component) + 1) * block, height),
        )
        for component in _connected_blocks(grid)
        if len(component) >= min_blocks
    ]
    return tuple(sorted(boxes, key=lambda b: (b.top, b.left)))


def _connected_blocks(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    """Four-connected components of a boolean block grid.

    Written out rather than pulled from scipy: the grid is small (a 4K frame
    at block 16 is 135x240) and adding an image-processing dependency for
    twenty lines is a poor trade.
    """
    rows, cols = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for row in range(rows):
        for col in range(cols):
            if not grid[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            component: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                component.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(component)
    return components


def frame_is_in_segment(frame: Frame, segments: tuple[TimeSegment, ...]) -> bool:
    """Whether a frame's timestamp falls in any segment."""
    return any(segment.contains(frame.timestamp_s) for segment in segments)
