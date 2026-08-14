"""Writing redacted output.

The writer is the only thing that turns a plan into bytes, and it is the only
thing that produces the digest those bytes hash to. That is deliberate: if the
digest were supplied by the caller, an attestation could be minted for media
nobody ever redacted, which is precisely the failure PLAN.md section 8's gate
exists to prevent. The pipeline hashes what the writer actually wrote.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from or_audit.media.frames import Frame, digest_file


@dataclass(frozen=True)
class WrittenOutput:
    """The result of writing redacted frames."""

    uri: str
    sha256: str
    frame_count: int
    frame_rate: float


@runtime_checkable
class FrameWriter(Protocol):
    """Persists redacted frames and reports what it wrote."""

    def write(self, frames: Iterable[Frame], *, frame_rate: float) -> WrittenOutput:
        """Write ``frames`` and return their locator and digest."""
        ...


class NpzFrameWriter:
    """Writes redacted frames to a compressed ``.npz`` stack.

    A frame stack rather than an encoded container: the alpha needs output
    that provably contains the redacted pixels and nothing else, and a lossless
    array dump gives that without pulling in an encoder or introducing
    compression artifacts that would make the digest depend on codec version.
    Swapping in an ffmpeg-backed writer is a matter of implementing the same
    protocol.
    """

    def __init__(self, path: Path) -> None:
        """Write to ``path``, which is created or overwritten."""
        self._path = path

    def write(self, frames: Iterable[Frame], *, frame_rate: float) -> WrittenOutput:
        """Write ``frames`` and hash the resulting file.

        Raises:
            ValueError: If no frames survive redaction. An empty output is
                never a valid de-identified recording, and writing one would
                let an episode pass the gate with nothing in it.
        """
        stack = [frame.pixels for frame in frames]
        if not stack:
            msg = (
                "redaction produced no frames; an empty output cannot be "
                "attested as a de-identified recording"
            )
            raise ValueError(msg)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._path,
            frames=np.stack(stack).astype(np.uint8),
            frame_rate=np.float64(frame_rate),
        )
        # numpy appends .npz when the path lacks the suffix.
        written = self._path if self._path.exists() else self._path.with_suffix(".npz")
        return WrittenOutput(
            uri=written.as_uri(),
            sha256=digest_file(written),
            frame_count=len(stack),
            frame_rate=frame_rate,
        )
