"""Detector behaviour on frames with known, constructed properties.

Synthetic frames rather than recordings: the detectors are deterministic
functions of pixel statistics, so a frame built to have a given redness ratio
or a given static region tests the logic exactly, with no licensing or PHI
problem. Real-footage validation is a clinical exercise, not a unit test, and
PLAN.md is explicit that these are screening heuristics rather than validated
classifiers.
"""

from __future__ import annotations

import numpy as np
import pytest

from or_audit.deid.detectors import (
    PixelBox,
    TimeSegment,
    detect_out_of_body,
    detect_static_overlays,
    redness_ratio,
)
from or_audit.media.frames import InMemoryFrameSource

HEIGHT, WIDTH = 64, 96


def in_body_frame(seed: int = 0) -> np.ndarray:
    """A red-dominated frame, as endoscopic imagery is."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[..., 0] = rng.integers(150, 210, (HEIGHT, WIDTH))
    frame[..., 1] = rng.integers(40, 80, (HEIGHT, WIDTH))
    frame[..., 2] = rng.integers(40, 80, (HEIGHT, WIDTH))
    return frame


def room_frame(seed: int = 0) -> np.ndarray:
    """A balanced-channel frame, as room imagery is."""
    rng = np.random.default_rng(seed)
    return rng.integers(90, 170, (HEIGHT, WIDTH, 3), dtype=np.uint8).astype(np.uint8)


class TestRednessRatio:
    def test_in_body_frame_is_red_dominated(self):
        assert redness_ratio(in_body_frame()) > 0.55

    def test_room_frame_is_near_achromatic(self):
        assert 0.30 < redness_ratio(room_frame()) < 0.38

    def test_the_two_classes_are_separated_by_the_default_threshold(self):
        """The detector's default only works if the gap actually exists."""
        assert redness_ratio(room_frame()) < 0.40 < redness_ratio(in_body_frame())

    def test_black_frame_is_treated_as_achromatic_not_in_body(self):
        """Undefined ratio must fail safe: flagged for review, not assumed clean."""
        assert redness_ratio(np.zeros((4, 4, 3), dtype=np.uint8)) == pytest.approx(1 / 3)

    def test_pure_red_saturates(self):
        pixels = np.zeros((4, 4, 3), dtype=np.uint8)
        pixels[..., 0] = 255
        assert redness_ratio(pixels) == pytest.approx(1.0)


class TestOutOfBodyDetection:
    def _source(self, pattern: str, *, frame_rate: float = 10.0) -> InMemoryFrameSource:
        """Build a source from a string: ``i`` in-body, ``o`` out-of-body."""
        frames = [
            in_body_frame(index) if char == "i" else room_frame(index)
            for index, char in enumerate(pattern)
        ]
        return InMemoryFrameSource(frames, frame_rate=frame_rate)

    def test_all_in_body_yields_no_segments(self):
        assert detect_out_of_body(self._source("i" * 40), stride=1) == ()

    def test_trailing_out_of_body_run_is_found(self):
        """Cameras are pulled out at the end of the case; that tail matters."""
        segments = detect_out_of_body(self._source("i" * 20 + "o" * 20), stride=1)
        assert len(segments) == 1
        assert segments[0].start_s == pytest.approx(2.0)
        assert segments[0].end_s == pytest.approx(4.0, abs=0.11)

    def test_mid_procedure_run_is_found(self):
        segments = detect_out_of_body(self._source("i" * 10 + "o" * 10 + "i" * 10), stride=1)
        assert len(segments) == 1
        assert segments[0].start_s == pytest.approx(1.0)
        assert segments[0].end_s == pytest.approx(2.0)

    def test_two_runs_are_reported_separately(self):
        pattern = "i" * 10 + "o" * 10 + "i" * 10 + "o" * 10
        assert len(detect_out_of_body(self._source(pattern), stride=1)) == 2

    def test_single_frame_flicker_is_suppressed(self):
        """One washed-out frame is not the camera leaving the body."""
        pattern = "i" * 10 + "o" + "i" * 10
        assert detect_out_of_body(self._source(pattern), stride=1, min_duration_s=0.5) == ()

    def test_flicker_is_reported_when_the_floor_is_removed(self):
        """Confirms suppression is the duration floor, not a detection failure."""
        pattern = "i" * 10 + "o" + "i" * 10
        assert len(detect_out_of_body(self._source(pattern), stride=1, min_duration_s=0.0)) == 1

    def test_final_frame_out_of_body_is_not_missed(self):
        """sample_indices always includes the last frame for exactly this case."""
        segments = detect_out_of_body(self._source("i" * 29 + "o"), stride=5, min_duration_s=0.0)
        assert segments
        assert segments[-1].end_s > 2.9

    def test_empty_source_yields_no_segments(self):
        assert detect_out_of_body(InMemoryFrameSource([], frame_rate=10.0)) == ()

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_threshold_rejected(self, bad):
        with pytest.raises(ValueError, match="threshold must be in"):
            detect_out_of_body(self._source("i" * 5), threshold=bad)


class TestStaticOverlayDetection:
    def _moving_source(
        self, *, overlay: tuple[int, int, int, int] | None = None
    ) -> InMemoryFrameSource:
        """Frames that change everywhere except an optional static block."""
        frames = []
        for index in range(12):
            frame = in_body_frame(index)
            if overlay is not None:
                left, top, right, bottom = overlay
                frame[top:bottom, left:right] = 255
            frames.append(frame)
        return InMemoryFrameSource(frames, frame_rate=10.0)

    def test_static_region_is_found(self):
        boxes = detect_static_overlays(self._moving_source(overlay=(0, 0, 32, 16)), stride=1)
        assert len(boxes) == 1
        assert boxes[0].left == 0
        assert boxes[0].top == 0
        assert boxes[0].right >= 32
        assert boxes[0].bottom >= 16

    def test_no_static_region_yields_nothing(self):
        assert detect_static_overlays(self._moving_source(), stride=1) == ()

    def test_two_separate_regions_are_reported_separately(self):
        frames = []
        for index in range(12):
            frame = in_body_frame(index)
            frame[0:16, 0:16] = 255
            frame[48:64, 80:96] = 200
            frames.append(frame)
        boxes = detect_static_overlays(InMemoryFrameSource(frames, frame_rate=10.0), stride=1)
        assert len(boxes) == 2

    def test_boxes_are_clamped_to_frame_bounds(self):
        """Block rounding must not produce a box outside the image."""
        boxes = detect_static_overlays(
            self._moving_source(overlay=(80, 48, 96, 64)), stride=1, block=16
        )
        assert all(b.right <= WIDTH and b.bottom <= HEIGHT for b in boxes)

    def test_single_sample_yields_nothing_rather_than_a_full_frame_box(self):
        """Variance over one frame is zero everywhere; that is not evidence."""
        source = InMemoryFrameSource([in_body_frame()], frame_rate=10.0)
        assert detect_static_overlays(source, stride=1) == ()

    def test_boxes_are_sorted_deterministically(self):
        frames = []
        for index in range(12):
            frame = in_body_frame(index)
            frame[48:64, 0:16] = 255
            frame[0:16, 64:80] = 255
            frames.append(frame)
        boxes = detect_static_overlays(InMemoryFrameSource(frames, frame_rate=10.0), stride=1)
        assert [(b.top, b.left) for b in boxes] == sorted((b.top, b.left) for b in boxes)


class TestGeometryTypes:
    def test_segment_requires_positive_duration(self):
        with pytest.raises(ValueError, match="positive duration"):
            TimeSegment(start_s=5.0, end_s=5.0)

    def test_segment_is_half_open(self):
        segment = TimeSegment(start_s=1.0, end_s=2.0)
        assert segment.contains(1.0)
        assert not segment.contains(2.0)

    def test_box_requires_positive_area(self):
        with pytest.raises(ValueError, match="positive area"):
            PixelBox(left=5, top=5, right=5, bottom=10)

    def test_box_rejects_negative_origin(self):
        with pytest.raises(ValueError, match="non-negative"):
            PixelBox(left=-1, top=0, right=5, bottom=5)
