"""Regression tests for defects found by adversarial review of Phase 2.

Each test here corresponds to a specific way PHI escaped, or a specific way an
attestation could be minted for work that did not happen. They are grouped
separately from the feature tests because their value is historical: these are
the mistakes actually made, and a refactor that reintroduces any of them is
not a style regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from or_audit.deid.detectors import detect_out_of_body
from or_audit.deid.pipeline import analyze, redact
from or_audit.deid.plan import PlannedSegment, RedactionPlan, apply_plan
from or_audit.deid.policy import DeidPolicy
from or_audit.deid.writer import NpzFrameWriter, WrittenOutput
from or_audit.domain.entities import MediaAsset
from or_audit.domain.enums import DeidStatus, MediaKind
from or_audit.domain.ids import new_episode_id, new_media_asset_id
from or_audit.errors import DeidentificationBoundaryError, DomainInvariantError
from or_audit.media.frames import Frame, InMemoryFrameSource

FRAME_RATE = 30.0


def in_body() -> np.ndarray:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 200, 40, 40
    return frame


def out_of_body() -> np.ndarray:
    return np.full((16, 16, 3), 130, dtype=np.uint8)


def is_out_of_body(frame: Frame) -> bool:
    """Out-of-body test frames are grey; in-body ones are red-dominated."""
    return float(frame.pixels[..., 0].mean()) < 180.0


def leaked_frame_count(pattern: list[np.ndarray], *, stride: int, **kwargs: float) -> int:
    """Redact ``pattern`` per the detector and count surviving out-of-body frames."""
    source = InMemoryFrameSource(pattern, frame_rate=FRAME_RATE)
    segments = detect_out_of_body(source, stride=stride, **kwargs)
    plan = RedactionPlan(
        policy_version="1",
        detectors=("redness-ratio@1",),
        source_frame_count=len(pattern),
        source_frame_rate=FRAME_RATE,
        dropped_segments=tuple(
            PlannedSegment(start_s=s.start_s, end_s=s.end_s, reason="exit") for s in segments
        ),
    )
    return sum(1 for frame in apply_plan(source, plan) if is_out_of_body(frame))


class TestOutOfBodyFramesDoNotSurvive:
    """A flagged sample must cover the unsampled gap on both sides.

    Original defect: coverage ran only forward from a flagged sample, so an
    exit beginning between two samples left every frame before the flagged one
    in the output. At stride 15, an exit at frame 8 leaked frames 8-14.
    """

    def test_mid_case_exit_between_samples_leaks_nothing(self):
        pattern = [in_body()] * 8 + [out_of_body()] * 22 + [in_body()] * 10
        assert leaked_frame_count(pattern, stride=15) == 0

    def test_exit_at_the_very_start_leaks_nothing(self):
        pattern = [out_of_body()] * 10 + [in_body()] * 30
        assert leaked_frame_count(pattern, stride=15) == 0

    @pytest.mark.parametrize("stride", [1, 3, 7, 15, 30])
    def test_no_stride_leaks(self, stride):
        """The gap grows with stride; coverage must grow with it."""
        pattern = [in_body()] * 40 + [out_of_body()] * 25 + [in_body()] * 25
        assert leaked_frame_count(pattern, stride=stride) == 0

    @pytest.mark.parametrize("onset", [1, 5, 8, 14, 16, 23, 44])
    def test_no_onset_offset_leaks(self, onset):
        """Sweep the exit across sample boundaries; none may leak."""
        pattern = [in_body()] * onset + [out_of_body()] * (90 - onset)
        assert leaked_frame_count(pattern, stride=15) == 0

    def test_clean_recording_is_left_alone(self):
        """Conservative expansion must not fire when there is no exit."""
        source = InMemoryFrameSource([in_body()] * 60, frame_rate=FRAME_RATE)
        assert detect_out_of_body(source, stride=15) == ()


class TestTailExitIsNeverSuppressed:
    """The duration floor must not reach the end of the recording.

    Original defect: a five-frame exit at 30fps spans 0.17s, below the 0.5s
    flicker floor, so the end-of-case camera withdrawal -- the most predictable
    out-of-body event there is -- was dropped from the plan entirely.
    """

    @pytest.mark.parametrize("tail", [1, 2, 5, 10])
    def test_short_tail_exit_is_still_redacted(self, tail):
        pattern = [in_body()] * (100 - tail) + [out_of_body()] * tail
        assert leaked_frame_count(pattern, stride=15) == 0

    def test_short_tail_exit_produces_a_segment(self):
        source = InMemoryFrameSource([in_body()] * 95 + [out_of_body()] * 5, frame_rate=FRAME_RATE)
        assert detect_out_of_body(source, stride=15) != ()

    def test_mid_case_flicker_is_still_suppressed(self):
        """Tail protection must not disable flicker suppression generally."""
        source = InMemoryFrameSource(
            [in_body()] * 40 + [out_of_body()] + [in_body()] * 40, frame_rate=FRAME_RATE
        )
        assert detect_out_of_body(source, stride=1, min_duration_s=0.5) == ()


class TestPlanIsBoundToItsSource:
    """A plan may only be applied to the material it was analysed on.

    Original defect: none. This closes an attack the review identified --
    analyse a clean short clip, obtain a no-op plan, then apply that plan to
    the real recording and attest the result.
    """

    def _asset(self) -> MediaAsset:
        return MediaAsset(
            id=new_media_asset_id(),
            episode_id=new_episode_id(),
            kind=MediaKind.ENDOSCOPIC_VIDEO,
            raw_uri="s3://raw/case.mp4",
            sha256="a" * 64,
            deid_status=DeidStatus.RAW,
        )

    def test_plan_from_a_different_frame_count_is_refused(self, tmp_path):
        # Overlays disabled so the short clip yields a genuinely empty plan:
        # three identical frames are entirely static, which the overlay
        # detector correctly flags. The attack being closed here is the
        # frame-count mismatch, so the plan needs to be a real no-op.
        policy = DeidPolicy(redact_overlays=False)
        clean = InMemoryFrameSource([in_body()] * 3, frame_rate=FRAME_RATE)
        dirty = InMemoryFrameSource([in_body()] * 60 + [out_of_body()] * 30, frame_rate=FRAME_RATE)
        analysed, noop_plan = analyze(self._asset(), clean, policy)
        assert noop_plan.is_noop
        with pytest.raises(DeidentificationBoundaryError, match="only be applied to"):
            redact(
                analysed,
                dirty,
                noop_plan,
                policy,
                NpzFrameWriter(tmp_path / "o.npz"),
                performed_by="deid-pipeline",
            )

    def test_plan_from_a_different_frame_rate_is_refused(self, tmp_path):
        frames = [in_body()] * 30
        analysed, plan = analyze(
            self._asset(), InMemoryFrameSource(frames, frame_rate=30.0), DeidPolicy()
        )
        with pytest.raises(DeidentificationBoundaryError, match="would not line up"):
            redact(
                analysed,
                InMemoryFrameSource(frames, frame_rate=60.0),
                plan,
                DeidPolicy(),
                NpzFrameWriter(tmp_path / "o.npz"),
                performed_by="deid-pipeline",
            )


class TestRedactRequiresAnalysis:
    """``redact`` must refuse a RAW asset.

    Original defect: RAW went straight to ATTESTED with no deid.started entry,
    so the trail showed a clearing with no analysis behind it -- and a caller
    holding a pre-analysis snapshot could mint a second attestation for media
    already attested through the proper path.
    """

    def test_raw_asset_cannot_be_redacted(self, tmp_path):
        source = InMemoryFrameSource([in_body()] * 30, frame_rate=FRAME_RATE)
        raw = MediaAsset(
            id=new_media_asset_id(),
            episode_id=new_episode_id(),
            kind=MediaKind.ENDOSCOPIC_VIDEO,
            raw_uri="s3://raw/case.mp4",
            sha256="a" * 64,
            deid_status=DeidStatus.RAW,
        )
        plan = RedactionPlan(
            policy_version="1",
            detectors=(),
            source_frame_count=30,
            source_frame_rate=FRAME_RATE,
        )
        with pytest.raises(DeidentificationBoundaryError, match="requires an analysed asset"):
            redact(
                raw,
                source,
                plan,
                DeidPolicy(),
                NpzFrameWriter(tmp_path / "o.npz"),
                performed_by="deid-pipeline",
            )

    def test_stale_pre_analysis_snapshot_cannot_mint_a_second_attestation(self, tmp_path):
        """The snapshot is RAW, so the ordering check catches the replay."""
        source = InMemoryFrameSource([in_body()] * 30, frame_rate=FRAME_RATE)
        raw = MediaAsset(
            id=new_media_asset_id(),
            episode_id=new_episode_id(),
            kind=MediaKind.ENDOSCOPIC_VIDEO,
            raw_uri="s3://raw/case.mp4",
            sha256="a" * 64,
            deid_status=DeidStatus.RAW,
        )
        analysed, plan = analyze(raw, source, DeidPolicy())
        redact(
            analysed,
            source,
            plan,
            DeidPolicy(),
            NpzFrameWriter(tmp_path / "first.npz"),
            performed_by="deid-pipeline",
        )
        with pytest.raises(DeidentificationBoundaryError):
            redact(
                raw,
                source,
                plan,
                DeidPolicy(),
                NpzFrameWriter(tmp_path / "second.npz"),
                performed_by="deid-pipeline",
            )


class TestWriterDigestIsNotTrusted:
    """The pipeline hashes the output itself.

    Original defect: ``FrameWriter`` is a protocol, so the digest it reported
    was an unverified assertion. A writer returning a fabricated digest
    produced an ATTESTED asset describing bytes nobody checked.
    """

    class _LyingWriter:
        """Writes real bytes but reports a digest of something else."""

        def __init__(self, path) -> None:
            self._inner = NpzFrameWriter(path)

        def write(self, frames, *, frame_rate: float) -> WrittenOutput:
            written = self._inner.write(frames, frame_rate=frame_rate)
            return WrittenOutput(
                uri=written.uri,
                sha256="f" * 64,
                frame_count=written.frame_count,
                frame_rate=written.frame_rate,
            )

    class _PhantomWriter:
        """Reports an output that does not exist."""

        def write(self, frames, *, frame_rate: float) -> WrittenOutput:
            list(frames)
            return WrittenOutput(
                uri="file:///nonexistent/phantom.npz",
                sha256="e" * 64,
                frame_count=1,
                frame_rate=frame_rate,
            )

    class _RemoteWriter:
        """Reports an output the pipeline cannot independently hash."""

        def write(self, frames, *, frame_rate: float) -> WrittenOutput:
            list(frames)
            return WrittenOutput(
                uri="s3://bucket/out.npz",
                sha256="d" * 64,
                frame_count=1,
                frame_rate=frame_rate,
            )

    def _analysed(self) -> tuple[MediaAsset, InMemoryFrameSource, RedactionPlan]:
        source = InMemoryFrameSource([in_body()] * 30, frame_rate=FRAME_RATE)
        raw = MediaAsset(
            id=new_media_asset_id(),
            episode_id=new_episode_id(),
            kind=MediaKind.ENDOSCOPIC_VIDEO,
            raw_uri="s3://raw/case.mp4",
            sha256="a" * 64,
            deid_status=DeidStatus.RAW,
        )
        analysed, plan = analyze(raw, source, DeidPolicy())
        return analysed, source, plan

    def test_fabricated_digest_is_caught(self, tmp_path):
        analysed, source, plan = self._analysed()
        with pytest.raises(DeidentificationBoundaryError, match="does not describe the bytes"):
            redact(
                analysed,
                source,
                plan,
                DeidPolicy(),
                self._LyingWriter(tmp_path / "o.npz"),
                performed_by="deid-pipeline",
            )

    def test_missing_output_file_is_caught(self):
        analysed, source, plan = self._analysed()
        with pytest.raises(DeidentificationBoundaryError, match="no file is there"):
            redact(
                analysed,
                source,
                plan,
                DeidPolicy(),
                self._PhantomWriter(),
                performed_by="deid-pipeline",
            )

    def test_unverifiable_remote_output_is_refused_not_trusted(self):
        """An unverifiable attestation is worse than none: it looks like evidence."""
        analysed, source, plan = self._analysed()
        with pytest.raises(DeidentificationBoundaryError, match="independently verify"):
            redact(
                analysed,
                source,
                plan,
                DeidPolicy(),
                self._RemoteWriter(),
                performed_by="deid-pipeline",
            )

    def test_honest_writer_still_works(self, tmp_path):
        """Re-hashes the file rather than comparing two internal fields.

        Asserting output_sha256 == final.sha256 alone is self-satisfying: a
        refactor that copied the writer's unverified digest into both fields
        would pass it. The disk hash is the only external reference.
        """
        import hashlib

        analysed, source, plan = self._analysed()
        final, attestation = redact(
            analysed,
            source,
            plan,
            DeidPolicy(),
            NpzFrameWriter(tmp_path / "o.npz"),
            performed_by="deid-pipeline",
        )
        on_disk = hashlib.sha256((tmp_path / "o.npz").read_bytes()).hexdigest()
        assert final.deid_status is DeidStatus.ATTESTED
        assert attestation.output_sha256 == on_disk
        assert final.sha256 == on_disk


class TestPolicyAndPlanVersionsMustAgree:
    """An attestation may not misdescribe which rules were in force."""

    def test_mismatched_versions_are_rejected(self):
        from datetime import UTC, datetime

        from or_audit.deid.attestation import DeidAttestation

        with pytest.raises(DomainInvariantError, match="policy version"):
            DeidAttestation(
                media_id=new_media_asset_id(),
                episode_id=new_episode_id(),
                media_kind=MediaKind.ENDOSCOPIC_VIDEO,
                performed_at=datetime(2026, 3, 4, tzinfo=UTC),
                performed_by="deid-pipeline",
                policy=DeidPolicy(version="1"),
                plan=RedactionPlan(
                    policy_version="99",
                    detectors=(),
                    source_frame_count=10,
                    source_frame_rate=FRAME_RATE,
                ),
                source_sha256="a" * 64,
                output_sha256="b" * 64,
                output_uri="file:///out.npz",
                output_frame_count=10,
            )


class TestSamplingRecallIsBoundedAndDeclared:
    """Sampling cannot silently lose short events.

    Original defect: `analysis_stride_frames` defaulted to 15, so an
    out-of-body run shorter than 15 frames could fall entirely between two
    samples and never be flagged. A 13-frame lens-clean at 30fps was invisible,
    and its frames were written into output attested as de-identified.

    Two changes close it. The default is now 1, so nothing is skipped. A caller
    who raises the stride still gets correct behaviour for events at or above
    the bound, and the plan records the bound so the artifact cannot be read as
    a completeness claim.
    """

    @pytest.mark.parametrize("run_length", [1, 2, 5, 8, 13, 14, 20])
    def test_short_runs_are_found_at_the_default_stride(self, run_length):
        pattern = [in_body()] * 16 + [out_of_body()] * run_length + [in_body()] * 100
        assert leaked_frame_count(pattern, stride=1) == 0

    @pytest.mark.parametrize("onset", [1, 8, 16, 17, 29, 31, 100, 101])
    def test_onset_sweep_leaks_nothing_at_the_default_stride(self, onset):
        pattern = [in_body()] * onset + [out_of_body()] * 13 + [in_body()] * 100
        assert leaked_frame_count(pattern, stride=1) == 0

    def test_default_policy_uses_full_recall(self):
        policy = DeidPolicy()
        assert policy.analysis_stride_frames == 1
        assert policy.out_of_body_min_duration_s == 0.0

    def test_plan_reports_no_recall_bound_at_full_sampling(self):
        source = InMemoryFrameSource([in_body()] * 60, frame_rate=FRAME_RATE)
        asset = MediaAsset(
            id=new_media_asset_id(),
            episode_id=new_episode_id(),
            kind=MediaKind.ENDOSCOPIC_VIDEO,
            raw_uri="s3://raw/case.mp4",
            sha256="a" * 64,
            deid_status=DeidStatus.RAW,
        )
        _, plan = analyze(asset, source, DeidPolicy())
        assert plan.is_recall_bounded is False
        assert plan.min_detectable_event_seconds == pytest.approx(1 / FRAME_RATE)

    def test_plan_declares_the_bound_when_a_caller_raises_the_stride(self):
        """Sampling is a legitimate speed trade, but not a silent one."""
        source = InMemoryFrameSource([in_body()] * 300, frame_rate=FRAME_RATE)
        asset = MediaAsset(
            id=new_media_asset_id(),
            episode_id=new_episode_id(),
            kind=MediaKind.ENDOSCOPIC_VIDEO,
            raw_uri="s3://raw/case.mp4",
            sha256="a" * 64,
            deid_status=DeidStatus.RAW,
        )
        _, plan = analyze(
            asset,
            source,
            DeidPolicy(analysis_stride_frames=15, sampling_justification="perf test"),
        )
        assert plan.is_recall_bounded is True
        assert plan.min_detectable_event_seconds == pytest.approx(0.5)

    def test_events_at_or_above_the_bound_are_still_found_when_sampling(self):
        """The bound is a real guarantee, not a disclaimer."""
        for onset in range(0, 30):
            pattern = [in_body()] * onset + [out_of_body()] * 15 + [in_body()] * 60
            assert leaked_frame_count(pattern, stride=15) == 0, f"onset {onset}"


class TestDurationFloorDoesNotDiscardRealExits:
    """The minimum-duration floor no longer suppresses true positives.

    Original defect: the floor defaulted to 0.5s to suppress single-frame
    flicker, but it cannot tell flicker from a genuine short exit. An 8-frame
    exit at 30fps spans 0.27s and was dropped from the plan entirely, so the
    room reached the attested output.
    """

    @pytest.mark.parametrize("run_length", [1, 2, 4, 8, 14])
    def test_sub_half_second_mid_case_exits_are_redacted(self, run_length):
        pattern = [in_body()] * 100 + [out_of_body()] * run_length + [in_body()] * 100
        assert leaked_frame_count(pattern, stride=1) == 0

    def test_floor_remains_available_to_callers_who_want_it(self):
        source = InMemoryFrameSource(
            [in_body()] * 100 + [out_of_body()] + [in_body()] * 100, frame_rate=FRAME_RATE
        )
        assert detect_out_of_body(source, stride=1, min_duration_s=0.5) == ()
        assert detect_out_of_body(source, stride=1) != ()


class TestTotalRedactionRoutesToDiscard:
    """A wholly out-of-body capture is destroyed, not attested as redacted."""

    def test_all_out_of_body_recording_is_refused_with_a_route(self, tmp_path):
        source = InMemoryFrameSource([out_of_body()] * 60, frame_rate=FRAME_RATE)
        asset = MediaAsset(
            id=new_media_asset_id(),
            episode_id=new_episode_id(),
            kind=MediaKind.ENDOSCOPIC_VIDEO,
            raw_uri="s3://raw/case.mp4",
            sha256="a" * 64,
            deid_status=DeidStatus.RAW,
        )
        analysed, plan = analyze(asset, source, DeidPolicy())
        assert plan.drops_everything
        with pytest.raises(DeidentificationBoundaryError, match="discard"):
            redact(
                analysed,
                source,
                plan,
                DeidPolicy(),
                NpzFrameWriter(tmp_path / "o.npz"),
                performed_by="deid-pipeline",
            )


class TestWriterPathResolution:
    """A non-.npz path must not resolve to the wrong file."""

    def test_non_npz_suffix_still_verifies(self, tmp_path):
        source = InMemoryFrameSource([in_body()] * 30, frame_rate=FRAME_RATE)
        asset = MediaAsset(
            id=new_media_asset_id(),
            episode_id=new_episode_id(),
            kind=MediaKind.ENDOSCOPIC_VIDEO,
            raw_uri="s3://raw/case.mp4",
            sha256="a" * 64,
            deid_status=DeidStatus.RAW,
        )
        analysed, plan = analyze(asset, source, DeidPolicy())
        final, attestation = redact(
            analysed,
            source,
            plan,
            DeidPolicy(),
            NpzFrameWriter(tmp_path / "out.bin"),
            performed_by="deid-pipeline",
        )
        assert final.deid_status is DeidStatus.ATTESTED
        assert (tmp_path / "out.bin.npz").exists()
        assert attestation.output_uri is not None
        assert attestation.output_uri.endswith("out.bin.npz")


class TestCoarseSamplingRequiresInformedConsent:
    """Bounded recall cannot be entered by accident.

    Residual leaks are only reachable by explicitly raising the stride above
    the guaranteed-detection bound. That is a legitimate speed trade on long
    recordings, but it is not a default and it is not silent: it mirrors the
    audio-retention rule, where departing from the safe disposition requires a
    recorded justification.
    """

    def test_raising_the_stride_without_a_justification_is_refused(self):
        """Asserts the domain error type, not merely 'some ValueError'.

        The looser assertion is what let the taxonomy leak ship: pydantic wraps
        a ValueError raised in a validator into ValidationError, which is a
        ValueError subclass, so `pytest.raises(ValueError)` passed while callers
        guarding on DeidentificationBoundaryError silently missed the rejection.
        """
        with pytest.raises(DeidentificationBoundaryError, match="requires sampling_justification"):
            DeidPolicy(analysis_stride_frames=15)

    def test_the_error_names_what_is_being_traded_away(self):
        with pytest.raises(
            DeidentificationBoundaryError, match="cannot detect out-of-body runs shorter"
        ):
            DeidPolicy(analysis_stride_frames=30)

    def test_policy_rejection_is_not_a_pydantic_validation_error(self):
        """Guard the boundary contract explicitly.

        ValidationError subclasses ValueError, so a test asserting ValueError
        cannot tell the two apart. This one can.
        """
        from pydantic import ValidationError

        with pytest.raises(DeidentificationBoundaryError) as caught:
            DeidPolicy(analysis_stride_frames=15)
        assert not isinstance(caught.value, ValidationError)

    def test_justified_coarse_sampling_is_permitted(self):
        policy = DeidPolicy(
            analysis_stride_frames=15,
            sampling_justification="8-hour archive backfill, reviewed by privacy office",
        )
        assert policy.analysis_stride_frames == 15

    def test_the_default_needs_no_justification(self):
        assert DeidPolicy().sampling_justification is None


class TestRedactionIsExactNotMerelySafe:
    """Coverage matches the gap, so no clinical footage is thrown away.

    Original defect: the span for a flagged sample ran from the *previous*
    sample to the *next* sample, so at stride 1 every exit destroyed one
    in-body frame on each side. Over-redaction fails safe, but it is still
    66ms of anatomy discarded for no information gain, and it was undocumented.

    A second, sharper defect hid inside the fix: computing the start as
    ``t(prev) + frame_period`` is not ``t(prev + 1)`` in binary floating point,
    and the discrepancy left the boundary frame outside its own half-open
    segment. That is a one-frame leak. Bounds are now derived from frame
    indices and divided once.
    """

    @pytest.mark.parametrize("onset", [0, 1, 8, 16, 17, 40, 100])
    @pytest.mark.parametrize("length", [1, 2, 4, 13])
    def test_exactly_the_out_of_body_frames_are_dropped(self, onset, length):
        total = 128
        pattern = [
            out_of_body() if onset <= i < onset + length else in_body() for i in range(total)
        ]
        source = InMemoryFrameSource(pattern, frame_rate=FRAME_RATE)
        segments = detect_out_of_body(source, stride=1)
        plan = RedactionPlan(
            policy_version="1",
            detectors=("redness-ratio@1",),
            source_frame_count=total,
            source_frame_rate=FRAME_RATE,
            dropped_segments=tuple(
                PlannedSegment(start_s=s.start_s, end_s=s.end_s, reason="exit") for s in segments
            ),
        )
        kept = list(apply_plan(source, plan))
        assert not [f for f in kept if is_out_of_body(f)], "out-of-body frame survived"
        assert len(kept) == total - length, "in-body frames were discarded unnecessarily"

    def test_clean_recording_is_passed_through_whole(self):
        pattern = [in_body()] * 128
        source = InMemoryFrameSource(pattern, frame_rate=FRAME_RATE)
        assert detect_out_of_body(source, stride=1) == ()

    def test_boundary_frame_is_inside_its_own_segment(self):
        """The float-summation bug showed up only at specific frame indices."""
        for onset in range(60, 70):
            pattern = [out_of_body() if onset <= i < onset + 25 else in_body() for i in range(90)]
            assert leaked_frame_count(pattern, stride=1) == 0, f"onset {onset}"
