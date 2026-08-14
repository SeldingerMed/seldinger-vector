"""Frame access.

De-identification and perception both need pixels, but neither should care
where the pixels came from. This module defines the boundary: a
:class:`FrameSource` yields RGB frames with timestamps, and everything
downstream is written against that protocol.

Two implementations ship here: an in-memory source (used by tests and by the
redaction pipeline, which works on decoded frames) and an ``.npz`` source that
reads the frame stacks the redaction writer produces. Decoding a real
container is an ffmpeg/PyAV adapter, deliberately not vendored into the core --
it is plumbing with a large dependency, and keeping it outside means the
detection logic stays testable without it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

#: Decoded RGB image, shape ``(height, width, 3)``, dtype ``uint8``.
RgbArray = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Frame:
    """One decoded frame.

    ``pixels`` is marked read-only on construction. Frames flow through
    detectors that have no business mutating them, and an accidental in-place
    write would corrupt every later consumer silently.
    """

    index: int
    timestamp_s: float
    pixels: RgbArray

    def __post_init__(self) -> None:
        if self.index < 0:
            msg = f"frame index must be non-negative, got {self.index}"
            raise ValueError(msg)
        if self.timestamp_s < 0:
            msg = f"frame timestamp must be non-negative, got {self.timestamp_s}"
            raise ValueError(msg)
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            msg = f"frame pixels must have shape (h, w, 3), got {self.pixels.shape}"
            raise ValueError(msg)
        if self.pixels.dtype != np.uint8:
            msg = f"frame pixels must be uint8, got {self.pixels.dtype}"
            raise ValueError(msg)
        self.pixels.flags.writeable = False

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.pixels.shape[1])


@runtime_checkable
class FrameSource(Protocol):
    """Read-only access to a sequence of frames."""

    @property
    def frame_count(self) -> int:
        """Total number of frames available."""
        ...

    @property
    def frame_rate(self) -> float:
        """Frames per second."""
        ...

    def read(self, index: int) -> Frame:
        """Return the frame at ``index``."""
        ...

    def iter_frames(self) -> Iterator[Frame]:
        """Iterate every frame in order."""
        ...


class InMemoryFrameSource:
    """A frame source backed by an in-memory stack."""

    def __init__(self, frames: Sequence[RgbArray], *, frame_rate: float) -> None:
        """Build a source from decoded frames.

        Args:
            frames: Frames in presentation order, each ``(h, w, 3)`` uint8.
            frame_rate: Frames per second. Must be positive; timestamps are
                derived from it, and a zero rate would make every frame share
                a timestamp.
        """
        if frame_rate <= 0:
            msg = f"frame_rate must be positive, got {frame_rate}"
            raise ValueError(msg)
        self._frames = list(frames)
        self._frame_rate = frame_rate

    @property
    def frame_count(self) -> int:
        """Total number of frames available."""
        return len(self._frames)

    @property
    def frame_rate(self) -> float:
        """Frames per second."""
        return self._frame_rate

    def read(self, index: int) -> Frame:
        """Return the frame at ``index``.

        Raises:
            IndexError: If ``index`` is out of range. Negative indices are
                rejected rather than wrapping, because a wrapped read during
                segment analysis would silently mix the end of a case into
                the beginning.
        """
        if not 0 <= index < len(self._frames):
            msg = f"frame index {index} out of range for {len(self._frames)} frames"
            raise IndexError(msg)
        return Frame(
            index=index,
            timestamp_s=index / self._frame_rate,
            pixels=self._frames[index],
        )

    def iter_frames(self) -> Iterator[Frame]:
        """Iterate every frame in order."""
        for index in range(len(self._frames)):
            yield self.read(index)


class NpzFrameSource(InMemoryFrameSource):
    """A frame source reading a stack written by the redaction writer."""

    def __init__(self, path: Path) -> None:
        """Load frames from an ``.npz`` archive.

        Args:
            path: Archive containing a ``frames`` array of shape
                ``(n, h, w, 3)`` and a scalar ``frame_rate``.
        """
        with np.load(path) as archive:
            stack = archive["frames"]
            frame_rate = float(archive["frame_rate"])
        super().__init__(list(stack.astype(np.uint8)), frame_rate=frame_rate)


def sample_indices(frame_count: int, *, stride: int) -> list[int]:
    """Evenly spaced frame indices for analysis.

    Detectors run on a sample rather than every frame: a 90-minute case at
    30fps is ~160k frames, and the properties being measured -- hue balance,
    temporal invariance -- are stable over far coarser sampling.

    Args:
        frame_count: Number of frames available.
        stride: Take every ``stride``-th frame.

    Returns:
        Indices in ascending order. Always includes the final frame, so a
        segment ending at the very end of a recording is not missed.
    """
    if stride < 1:
        msg = f"stride must be at least 1, got {stride}"
        raise ValueError(msg)
    if frame_count <= 0:
        return []
    indices = list(range(0, frame_count, stride))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def digest_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 of a file's bytes."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
