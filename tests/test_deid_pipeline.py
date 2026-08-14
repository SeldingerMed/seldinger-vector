"""The de-identification gate, end to end.

The property under test throughout: an asset reaches ``ATTESTED`` only after
bytes were written and hashed by the pipeline. Everything else in PLAN.md
section 8 rests on that, because a status field that can be set by assertion
protects nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from or_audit.audit.trail import Actor, ActorKind, AuditAction, AuditTrail
from or_audit.deid.attestation import DeidAttestation
from or_audit.deid.pipeline import analyze, default_disposition, discard, redact
from or_audit.deid.plan import PlannedBox, PlannedSegment, RedactionPlan, apply_plan
from or_audit.deid.policy import AudioDisposition, DeidPolicy
from or_audit.deid.writer import NpzFrameWriter
from or_audit.domain.entities import MediaAsset
from or_audit.domain.enums import DeidStatus, MediaKind
from or_audit.domain.ids import new_episode_id, new_media_asset_id
from or_audit.errors import DeidentificationBoundaryError, DomainInvariantError
from or_audit.media.frames import InMemoryFrameSource, NpzFrameSource

from .test_deid_detectors import in_body_frame, room_frame

CLOCK = datetime(2026, 3, 4, 15, 0, tzinfo=UTC)


@pytest.fixture
def policy() -> DeidPolicy:
    """A policy cleared to attest.

    Attestation requires a recorded overlay-bound measurement (PLAN.md V-10);
    the bare default deliberately cannot attest. These tests exercise the
    attestation mechanics, so they use a validated policy and the gate itself is
    covered in tests/test_deid_leaks.py.
    """
    return DeidPolicy(
        overlay_bound_validated_against="test fixture: synthetic overlay 16px vs 8px bound"
    )


@pytest.fixture
def episode_id() -> str:
    return new_episode_id()


def make_asset(episode_id: str, kind: MediaKind = MediaKind.ENDOSCOPIC_VIDEO) -> MediaAsset:
    return MediaAsset(
        id=new_media_asset_id(),
        episode_id=episode_id,
        kind=kind,
        raw_uri="s3://raw/case.mp4",
        sha256="a" * 64,
        deid_status=DeidStatus.RAW,
    )


#: Ninety frames at 30fps, i.e. three seconds. Long enough that the default
#: 15-frame analysis stride resolves the exit without the conservative
#: both-sides gap expansion swallowing the whole clip. On a 20-frame clip it
#: does swallow everything, and that is correct behaviour rather than a bug:
#: at stride 15 the detector genuinely cannot localise the transition.
DIRTY_FRAME_COUNT = 90
DIRTY_FRAME_RATE = 30.0


def dirty_source(*, overlay: bool = True, tail_out_of_body: bool = True) -> InMemoryFrameSource:
    """An in-body recording, optionally with an overlay and a closing exit."""
    frames = []
    for index in range(DIRTY_FRAME_COUNT):
        frame = room_frame(index) if tail_out_of_body and index >= 75 else in_body_frame(index)
        if overlay:
            frame[0:16, 0:32] = 255
        frames.append(frame)
    return InMemoryFrameSource(frames, frame_rate=DIRTY_FRAME_RATE)


@pytest.fixture
def trail() -> AuditTrail:
    return AuditTrail()


@pytest.fixture
def actor() -> Actor:
    return Actor(kind=ActorKind.SERVICE, ref="deid-pipeline")


class TestAnalyzeDoesNotClear:
    def test_analyze_moves_to_in_progress_not_attested(self, episode_id, policy):
        updated, _ = analyze(make_asset(episode_id), dirty_source(), policy)
        assert updated.deid_status is DeidStatus.IN_PROGRESS
        assert updated.is_readable is False

    def test_analysed_asset_still_fails_the_read_gate(self, episode_id, policy):
        updated, _ = analyze(make_asset(episode_id), dirty_source(), policy)
        with pytest.raises(DeidentificationBoundaryError):
            _ = updated.readable_uri

    def test_plan_records_what_ran(self, episode_id, policy):
        _, plan = analyze(make_asset(episode_id), dirty_source(), policy)
        assert plan.policy_version == policy.version
        assert any("redness-ratio" in d for d in plan.detectors)
        assert any("temporal-invariance" in d for d in plan.detectors)
        assert plan.dropped_segments
        assert plan.masked_boxes

    def test_clean_recording_yields_a_noop_plan(self, episode_id, policy):
        source = dirty_source(overlay=False, tail_out_of_body=False)
        _, plan = analyze(make_asset(episode_id), source, policy)
        assert plan.is_noop

    def test_disabled_detectors_are_not_recorded(self, episode_id):
        policy = DeidPolicy(redact_overlays=False)
        _, plan = analyze(make_asset(episode_id), dirty_source(), policy)
        assert not any("temporal-invariance" in d for d in plan.detectors)
        assert plan.masked_boxes == ()

    def test_zero_frame_source_is_rejected(self, episode_id, policy):
        with pytest.raises(ValueError, match="zero frames"):
            analyze(make_asset(episode_id), InMemoryFrameSource([], frame_rate=10.0), policy)

    def test_audio_cannot_be_frame_analysed(self, episode_id, policy):
        with pytest.raises(ValueError, match="frame analysis applies to video"):
            analyze(make_asset(episode_id, MediaKind.AUDIO), dirty_source(), policy)


class TestRedactProducesRealOutput:
    def test_attested_only_after_bytes_are_written(self, episode_id, policy, tmp_path):
        asset = make_asset(episode_id)
        analysed, plan = analyze(asset, dirty_source(), policy)
        writer = NpzFrameWriter(tmp_path / "out.npz")
        final, attestation = redact(
            analysed, dirty_source(), plan, policy, writer, performed_by="deid-pipeline"
        )
        assert final.deid_status is DeidStatus.ATTESTED
        assert (tmp_path / "out.npz").exists()
        assert attestation.output_sha256 is not None

    def test_recorded_digest_is_the_digest_of_the_file_on_disk(self, episode_id, policy, tmp_path):
        """The whole gate rests on this: the hash describes real bytes."""
        import hashlib

        analysed, plan = analyze(make_asset(episode_id), dirty_source(), policy)
        writer = NpzFrameWriter(tmp_path / "out.npz")
        final, attestation = redact(
            analysed, dirty_source(), plan, policy, writer, performed_by="deid-pipeline"
        )
        on_disk = hashlib.sha256((tmp_path / "out.npz").read_bytes()).hexdigest()
        assert attestation.output_sha256 == on_disk
        assert final.sha256 == on_disk

    def test_asset_points_at_the_attestation_that_justifies_it(self, episode_id, policy, tmp_path):
        analysed, plan = analyze(make_asset(episode_id), dirty_source(), policy)
        final, attestation = redact(
            analysed,
            dirty_source(),
            plan,
            policy,
            NpzFrameWriter(tmp_path / "out.npz"),
            performed_by="deid-pipeline",
        )
        assert final.deid_attestation_sha256 == attestation.digest

    def test_output_no_longer_contains_the_out_of_body_frames(self, episode_id, policy, tmp_path):
        """Dropped, not blanked: the material must be absent from the file."""
        analysed, plan = analyze(make_asset(episode_id), dirty_source(), policy)
        redact(
            analysed,
            dirty_source(),
            plan,
            policy,
            NpzFrameWriter(tmp_path / "out.npz"),
            performed_by="deid-pipeline",
        )
        reloaded = NpzFrameSource(tmp_path / "out.npz")
        assert reloaded.frame_count < DIRTY_FRAME_COUNT

    def test_output_has_the_overlay_region_zeroed(self, episode_id, policy, tmp_path):
        analysed, plan = analyze(make_asset(episode_id), dirty_source(), policy)
        redact(
            analysed,
            dirty_source(),
            plan,
            policy,
            NpzFrameWriter(tmp_path / "out.npz"),
            performed_by="deid-pipeline",
        )
        reloaded = NpzFrameSource(tmp_path / "out.npz")
        assert reloaded.read(0).pixels[0:16, 0:32].max() == 0

    def test_readable_after_attestation(self, episode_id, policy, tmp_path):
        analysed, plan = analyze(make_asset(episode_id), dirty_source(), policy)
        final, _ = redact(
            analysed,
            dirty_source(),
            plan,
            policy,
            NpzFrameWriter(tmp_path / "out.npz"),
            performed_by="deid-pipeline",
        )
        assert final.readable_uri.endswith("out.npz")

    def test_total_redaction_is_routed_to_discard(self, episode_id, policy, tmp_path):
        """A wholly out-of-body capture is destroyed, not attested as redacted."""
        analysed, _ = analyze(make_asset(episode_id), dirty_source(), policy)
        drop_all = RedactionPlan(
            policy_version=policy.version,
            detectors=(),
            source_frame_count=DIRTY_FRAME_COUNT,
            source_frame_rate=DIRTY_FRAME_RATE,
            dropped_segments=(PlannedSegment(start_s=0.0, end_s=99.0, reason="everything"),),
        )
        with pytest.raises(DeidentificationBoundaryError, match="discard\\(\\) rather than"):
            redact(
                analysed,
                dirty_source(),
                drop_all,
                policy,
                NpzFrameWriter(tmp_path / "out.npz"),
                performed_by="deid-pipeline",
            )


class TestSettledMediaIsNotReprocessed:
    @pytest.mark.parametrize("status", [DeidStatus.ATTESTED, DeidStatus.DISCARDED])
    def test_analyze_refuses_settled_media(self, episode_id, policy, status):
        asset = make_asset(episode_id).model_copy(
            update={
                "deid_status": status,
                "deid_attestation_sha256": "b" * 64 if status is DeidStatus.ATTESTED else None,
            }
        )
        with pytest.raises(DeidentificationBoundaryError, match="already"):
            analyze(asset, dirty_source(), policy)

    def test_discard_refuses_settled_media(self, episode_id, policy):
        asset = make_asset(episode_id, MediaKind.AUDIO).model_copy(
            update={"deid_status": DeidStatus.DISCARDED}
        )
        with pytest.raises(DeidentificationBoundaryError, match="already"):
            discard(asset, policy, reason="again", performed_by="op-1")


class TestDiscard:
    def test_audio_is_discarded_by_default_policy(self, episode_id, policy):
        asset = make_asset(episode_id, MediaKind.AUDIO)
        reason = default_disposition(asset, policy)
        assert reason is not None
        updated, attestation = discard(asset, policy, reason=reason, performed_by="deid-pipeline")
        assert updated.deid_status is DeidStatus.DISCARDED
        assert attestation.discarded is True
        assert attestation.output_sha256 is None

    def test_room_video_is_discarded_by_default_policy(self, episode_id, policy):
        asset = make_asset(episode_id, MediaKind.ROOM_VIDEO)
        assert default_disposition(asset, policy) is not None

    def test_endoscopic_video_is_never_discarded_by_default(self, episode_id, policy):
        assert default_disposition(make_asset(episode_id), policy) is None

    def test_retaining_audio_requires_a_justification(self):
        with pytest.raises(
            DeidentificationBoundaryError, match="requires audio_retention_justification"
        ):
            DeidPolicy(audio=AudioDisposition.RETAIN_WITH_REVIEW)

    def test_retained_audio_is_not_auto_discarded(self, episode_id):
        policy = DeidPolicy(
            audio=AudioDisposition.RETAIN_WITH_REVIEW,
            audio_retention_justification="IRB-approved phonosurgery study",
        )
        assert default_disposition(make_asset(episode_id, MediaKind.AUDIO), policy) is None


class TestAttestationInvariants:
    def _base(self, episode_id, policy, **overrides: object) -> DeidAttestation:
        payload = {
            "media_id": new_media_asset_id(),
            "episode_id": episode_id,
            "media_kind": MediaKind.ENDOSCOPIC_VIDEO,
            "performed_at": CLOCK,
            "performed_by": "deid-pipeline",
            "policy": policy,
            "plan": RedactionPlan(
                policy_version=policy.version,
                detectors=(),
                source_frame_count=10,
                source_frame_rate=10.0,
            ),
            "source_sha256": "a" * 64,
            "output_sha256": "b" * 64,
            "output_uri": "file:///out.npz",
            "output_frame_count": 10,
        }
        return DeidAttestation(**(payload | overrides))

    def test_valid_attestation_builds(self, episode_id, policy):
        assert self._base(episode_id, policy).digest

    def test_non_discarded_must_name_an_output(self, episode_id, policy):
        with pytest.raises(DomainInvariantError, match="must name"):
            self._base(episode_id, policy, output_sha256=None, output_uri=None)

    def test_discarded_must_not_name_an_output(self, episode_id, policy):
        with pytest.raises(DomainInvariantError, match="names an output"):
            self._base(episode_id, policy, discarded=True, discard_reason="policy")

    def test_discarded_needs_a_reason(self, episode_id, policy):
        with pytest.raises(DomainInvariantError, match="without a reason"):
            self._base(
                episode_id,
                policy,
                discarded=True,
                output_sha256=None,
                output_uri=None,
                output_frame_count=None,
            )

    def test_claiming_redaction_with_an_unchanged_output_is_rejected(self, episode_id, policy):
        """The clearest signature of a fake attestation."""
        plan = RedactionPlan(
            policy_version=policy.version,
            detectors=("x@1",),
            source_frame_count=10,
            source_frame_rate=10.0,
            masked_boxes=(PlannedBox(left=0, top=0, right=4, bottom=4, reason="overlay"),),
        )
        with pytest.raises(DomainInvariantError, match="byte-identical"):
            self._base(episode_id, policy, plan=plan, output_sha256="a" * 64)

    def test_digest_changes_when_any_field_changes(self, episode_id, policy):
        first = self._base(episode_id, policy)
        second = self._base(episode_id, policy, output_frame_count=9)
        assert first.digest != second.digest


class TestApplyPlan:
    def test_masked_region_is_zeroed_and_the_rest_survives(self):
        frames = [np.full((8, 8, 3), 200, dtype=np.uint8) for _ in range(4)]
        source = InMemoryFrameSource(frames, frame_rate=4.0)
        plan = RedactionPlan(
            policy_version="1",
            detectors=(),
            source_frame_count=4,
            source_frame_rate=4.0,
            masked_boxes=(PlannedBox(left=0, top=0, right=4, bottom=4, reason="overlay"),),
        )
        out = list(apply_plan(source, plan))
        assert out[0].pixels[0:4, 0:4].max() == 0
        assert out[0].pixels[4:8, 4:8].min() == 200

    def test_timestamps_are_rebased_after_a_drop(self):
        """A gap in the output timeline would make every later duration wrong."""
        frames = [np.full((4, 4, 3), 10, dtype=np.uint8) for _ in range(6)]
        source = InMemoryFrameSource(frames, frame_rate=2.0)
        plan = RedactionPlan(
            policy_version="1",
            detectors=(),
            source_frame_count=6,
            source_frame_rate=2.0,
            dropped_segments=(PlannedSegment(start_s=0.0, end_s=1.0, reason="exit"),),
        )
        out = list(apply_plan(source, plan))
        assert [f.index for f in out] == [0, 1, 2, 3]
        assert [f.timestamp_s for f in out] == [0.0, 0.5, 1.0, 1.5]

    def test_noop_plan_passes_frames_through(self):
        frames = [np.full((4, 4, 3), 7, dtype=np.uint8) for _ in range(3)]
        plan = RedactionPlan(
            policy_version="1", detectors=(), source_frame_count=3, source_frame_rate=1.0
        )
        out = list(apply_plan(InMemoryFrameSource(frames, frame_rate=1.0), plan))
        assert len(out) == 3
        assert out[0].pixels.min() == 7

    def test_source_frames_are_not_mutated(self):
        """Masking must copy; the caller's decoded frames are shared."""
        original = np.full((8, 8, 3), 200, dtype=np.uint8)
        source = InMemoryFrameSource([original.copy()], frame_rate=1.0)
        plan = RedactionPlan(
            policy_version="1",
            detectors=(),
            source_frame_count=1,
            source_frame_rate=1.0,
            masked_boxes=(PlannedBox(left=0, top=0, right=4, bottom=4, reason="overlay"),),
        )
        list(apply_plan(source, plan))
        assert source.read(0).pixels[0:4, 0:4].min() == 200


class TestAuditIntegration:
    def test_full_run_records_the_transitions(self, episode_id, policy, tmp_path, trail, actor):
        asset = make_asset(episode_id)
        analysed, plan = analyze(asset, dirty_source(), policy, trail=trail, actor=actor)
        redact(
            analysed,
            dirty_source(),
            plan,
            policy,
            NpzFrameWriter(tmp_path / "out.npz"),
            performed_by="deid-pipeline",
            trail=trail,
            actor=actor,
        )
        assert [e.action for e in trail] == [
            AuditAction.DEID_STARTED,
            AuditAction.DEID_ATTESTED,
        ]
        trail.verify()

    def test_attested_entry_carries_the_attestation_digest(
        self, episode_id, policy, tmp_path, trail, actor
    ):
        analysed, plan = analyze(make_asset(episode_id), dirty_source(), policy)
        _, attestation = redact(
            analysed,
            dirty_source(),
            plan,
            policy,
            NpzFrameWriter(tmp_path / "out.npz"),
            performed_by="deid-pipeline",
            trail=trail,
            actor=actor,
        )
        assert trail.entries[0].payload["attestation_sha256"] == attestation.digest

    def test_trail_without_actor_is_rejected(self, episode_id, policy, trail):
        with pytest.raises(ValueError, match="without an actor"):
            analyze(make_asset(episode_id), dirty_source(), policy, trail=trail)
