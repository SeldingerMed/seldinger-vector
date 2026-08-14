"""Frame model and source contracts."""

from __future__ import annotations

import numpy as np
import pytest

from or_audit.media.frames import (
    Frame,
    FrameSource,
    InMemoryFrameSource,
    NpzFrameSource,
    digest_file,
    sample_indices,
)


def rgb(value: int = 128, *, h: int = 4, w: int = 6) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestFrameValidation:
    def test_valid_frame_builds(self):
        frame = Frame(index=0, timestamp_s=0.0, pixels=rgb())
        assert frame.height == 4
        assert frame.width == 6

    def test_pixels_are_read_only(self):
        """Detectors must not be able to corrupt shared decoded frames."""
        frame = Frame(index=0, timestamp_s=0.0, pixels=rgb())
        with pytest.raises(ValueError, match="read-only"):
            frame.pixels[0, 0, 0] = 255

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError, match="index must be non-negative"):
            Frame(index=-1, timestamp_s=0.0, pixels=rgb())

    def test_negative_timestamp_rejected(self):
        with pytest.raises(ValueError, match="timestamp must be non-negative"):
            Frame(index=0, timestamp_s=-0.1, pixels=rgb())

    @pytest.mark.parametrize("shape", [(4, 6), (4, 6, 1), (4, 6, 4), (3, 4, 6, 3)])
    def test_non_rgb_shape_rejected(self, shape):
        with pytest.raises(ValueError, match=r"shape \(h, w, 3\)"):
            Frame(index=0, timestamp_s=0.0, pixels=np.zeros(shape, dtype=np.uint8))

    def test_non_uint8_dtype_rejected(self):
        """Float frames would silently change every detector threshold."""
        with pytest.raises(ValueError, match="must be uint8"):
            Frame(index=0, timestamp_s=0.0, pixels=np.zeros((4, 6, 3), dtype=np.float32))


class TestInMemoryFrameSource:
    def test_satisfies_the_protocol(self):
        assert isinstance(InMemoryFrameSource([rgb()], frame_rate=1.0), FrameSource)

    def test_timestamps_derive_from_the_frame_rate(self):
        source = InMemoryFrameSource([rgb(), rgb(), rgb()], frame_rate=4.0)
        assert [f.timestamp_s for f in source.iter_frames()] == [0.0, 0.25, 0.5]

    def test_frame_count_and_rate_are_reported(self):
        source = InMemoryFrameSource([rgb(), rgb()], frame_rate=8.0)
        assert source.frame_count == 2
        assert source.frame_rate == 8.0

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_frame_rate_rejected(self, bad):
        with pytest.raises(ValueError, match="frame_rate must be positive"):
            InMemoryFrameSource([rgb()], frame_rate=bad)

    def test_out_of_range_read_raises(self):
        with pytest.raises(IndexError, match="out of range"):
            InMemoryFrameSource([rgb()], frame_rate=1.0).read(5)

    def test_negative_index_does_not_wrap(self):
        """Wrapping would mix the end of a case into its beginning."""
        with pytest.raises(IndexError, match="out of range"):
            InMemoryFrameSource([rgb(1), rgb(2)], frame_rate=1.0).read(-1)

    def test_empty_source_is_permitted_and_iterates_empty(self):
        source = InMemoryFrameSource([], frame_rate=1.0)
        assert source.frame_count == 0
        assert list(source.iter_frames()) == []


class TestSampleIndices:
    def test_stride_one_takes_everything(self):
        assert sample_indices(5, stride=1) == [0, 1, 2, 3, 4]

    def test_final_frame_is_always_included(self):
        """A segment ending at the very end of a recording must not be missed."""
        assert sample_indices(10, stride=4) == [0, 4, 8, 9]

    def test_no_duplicate_when_stride_divides_evenly(self):
        assert sample_indices(9, stride=4) == [0, 4, 8]

    def test_single_frame(self):
        assert sample_indices(1, stride=10) == [0]

    def test_zero_frames_yields_nothing(self):
        assert sample_indices(0, stride=3) == []

    @pytest.mark.parametrize("bad", [0, -1])
    def test_invalid_stride_rejected(self, bad):
        with pytest.raises(ValueError, match="stride must be at least 1"):
            sample_indices(10, stride=bad)


class TestNpzRoundTrip:
    def test_frames_survive_a_write_and_reload(self, tmp_path):
        from or_audit.deid.writer import NpzFrameWriter

        frames = [Frame(index=i, timestamp_s=i / 5.0, pixels=rgb(10 * i + 1)) for i in range(3)]
        written = NpzFrameWriter(tmp_path / "s.npz").write(frames, frame_rate=5.0)
        reloaded = NpzFrameSource(tmp_path / "s.npz")
        assert reloaded.frame_count == 3
        assert reloaded.frame_rate == 5.0
        assert reloaded.read(1).pixels.min() == 11
        assert written.frame_count == 3

    def test_reported_digest_matches_the_file(self, tmp_path):
        from or_audit.deid.writer import NpzFrameWriter

        frames = [Frame(index=0, timestamp_s=0.0, pixels=rgb())]
        written = NpzFrameWriter(tmp_path / "s.npz").write(frames, frame_rate=1.0)
        assert written.sha256 == digest_file(tmp_path / "s.npz")

    def test_writer_creates_missing_parent_directories(self, tmp_path):
        from or_audit.deid.writer import NpzFrameWriter

        target = tmp_path / "nested" / "deeper" / "s.npz"
        NpzFrameWriter(target).write(
            [Frame(index=0, timestamp_s=0.0, pixels=rgb())], frame_rate=1.0
        )
        assert target.exists()


class TestDigestFile:
    def test_digest_is_stable_and_content_dependent(self, tmp_path):
        first = tmp_path / "a.bin"
        second = tmp_path / "b.bin"
        first.write_bytes(b"hello")
        second.write_bytes(b"hello")
        assert digest_file(first) == digest_file(second)
        second.write_bytes(b"hellp")
        assert digest_file(first) != digest_file(second)

    def test_large_file_is_chunked_correctly(self, tmp_path):
        import hashlib

        payload = bytes(range(256)) * 8192  # 2 MiB, crosses the chunk boundary
        path = tmp_path / "big.bin"
        path.write_bytes(payload)
        assert digest_file(path) == hashlib.sha256(payload).hexdigest()
