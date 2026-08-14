"""The de-identification pipeline.

Two steps, deliberately separate:

``analyze``
    Runs detectors and produces a :class:`RedactionPlan`. Changes nothing and
    clears nothing. The asset moves to ``IN_PROGRESS``.
``redact``
    Applies the plan, writes output, hashes the bytes it wrote, and only then
    issues an attestation and moves the asset to ``ATTESTED``.

The split exists so a plan can be reviewed before bytes are written, and so
there is no path from "we looked at it" to "it is clean". The digest on the
attestation is computed here from the writer's output; it is never accepted
from a caller. Without that rule the ``ATTESTED`` gate could be satisfied by
assertion, which would make PLAN.md section 8 decorative.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from or_audit.audit.trail import Actor, AuditAction, AuditTrail
from or_audit.deid.attestation import DeidAttestation
from or_audit.deid.detectors import (
    OUT_OF_BODY_DETECTOR,
    OUT_OF_BODY_DETECTOR_VERSION,
    OVERLAY_DETECTOR,
    OVERLAY_DETECTOR_VERSION,
    detect_out_of_body,
    detect_static_overlays,
)
from or_audit.deid.plan import PlannedBox, PlannedSegment, RedactionPlan, apply_plan
from or_audit.deid.policy import AudioDisposition, DeidPolicy
from or_audit.deid.writer import FrameWriter, WrittenOutput
from or_audit.domain.entities import MediaAsset
from or_audit.domain.enums import DeidStatus, MediaKind
from or_audit.errors import DeidentificationBoundaryError
from or_audit.media.frames import FrameSource, digest_file

_VIDEO_KINDS = frozenset({MediaKind.ENDOSCOPIC_VIDEO, MediaKind.ROOM_VIDEO})


def _utc_now() -> datetime:
    return datetime.now(UTC)


def analyze(
    asset: MediaAsset,
    source: FrameSource,
    policy: DeidPolicy,
    *,
    trail: AuditTrail | None = None,
    actor: Actor | None = None,
) -> tuple[MediaAsset, RedactionPlan]:
    """Detect PHI risks and produce a redaction plan.

    Args:
        asset: The media to analyse. Must not already be attested.
        source: Decoded frames for ``asset``.
        policy: Rules in force.
        trail: Optional audit trail to record the transition on.
        actor: Required when ``trail`` is supplied.

    Returns:
        The asset moved to ``IN_PROGRESS``, and the plan.

    Raises:
        DeidentificationBoundaryError: If the asset is already attested or
            discarded. Re-analysing settled media would let a second run
            silently replace a prior attestation.
        ValueError: If the asset is not video.
    """
    _reject_settled(asset)
    if asset.kind not in _VIDEO_KINDS:
        msg = f"media {asset.id} is {asset.kind.value}; frame analysis applies to video only"
        raise ValueError(msg)
    if source.frame_count == 0:
        msg = f"media {asset.id} decoded to zero frames; nothing to analyse"
        raise ValueError(msg)

    detectors: list[str] = []
    segments: tuple[PlannedSegment, ...] = ()
    boxes: tuple[PlannedBox, ...] = ()

    if policy.redact_out_of_body:
        detectors.append(f"{OUT_OF_BODY_DETECTOR}@{OUT_OF_BODY_DETECTOR_VERSION}")
        segments = tuple(
            PlannedSegment(
                start_s=found.start_s,
                end_s=found.end_s,
                reason="camera appears to be outside the patient",
            )
            for found in detect_out_of_body(
                source,
                threshold=policy.out_of_body_threshold,
                stride=policy.analysis_stride_frames,
                min_duration_s=policy.out_of_body_min_duration_s,
            )
        )

    if policy.redact_overlays:
        detectors.append(f"{OVERLAY_DETECTOR}@{OVERLAY_DETECTOR_VERSION}")
        boxes = tuple(
            PlannedBox(
                left=found.left,
                top=found.top,
                right=found.right,
                bottom=found.bottom,
                reason="region is temporally static and may carry burned-in identifiers",
            )
            for found in detect_static_overlays(
                source,
                stride=policy.overlay_stride_frames,
                max_std=policy.overlay_max_std,
                block=policy.overlay_block_px,
            )
        )

    plan = RedactionPlan(
        policy_version=policy.version,
        detectors=tuple(detectors),
        source_frame_count=source.frame_count,
        source_frame_rate=source.frame_rate,
        analysis_stride_frames=policy.analysis_stride_frames,
        dropped_segments=segments,
        masked_boxes=boxes,
    )
    updated = asset.model_copy(update={"deid_status": DeidStatus.IN_PROGRESS})
    _record(
        trail,
        actor,
        AuditAction.DEID_STARTED,
        asset.episode_id,
        {
            "media_id": asset.id,
            "detectors": list(plan.detectors),
            "dropped_segments": len(plan.dropped_segments),
            "masked_boxes": len(plan.masked_boxes),
            "min_detectable_event_seconds": plan.min_detectable_event_seconds,
        },
    )
    return updated, plan


def redact(
    asset: MediaAsset,
    source: FrameSource,
    plan: RedactionPlan,
    policy: DeidPolicy,
    writer: FrameWriter,
    *,
    performed_by: str,
    trail: AuditTrail | None = None,
    actor: Actor | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> tuple[MediaAsset, DeidAttestation]:
    """Apply ``plan``, write the output, and attest to what was produced.

    The output digest is computed from the bytes the writer wrote. There is no
    parameter for supplying it.

    Args:
        asset: Asset under analysis.
        source: Decoded source frames.
        plan: Plan from :func:`analyze`.
        policy: Rules in force.
        writer: Destination for the redacted frames.
        performed_by: Pseudonymous handle of the responsible principal.
        trail: Optional audit trail.
        actor: Required when ``trail`` is supplied.
        clock: Injectable time source.

    Returns:
        The asset moved to ``ATTESTED`` with the attestation digest recorded,
        and the attestation itself.

    Raises:
        DeidentificationBoundaryError: If the asset is already settled, if it
            has not been analysed, if the plan was not built from this source,
            or if the writer's reported digest does not match its output.
    """
    _reject_settled(asset)
    _require_analysed(asset)
    _require_plan_matches_source(asset, plan, source)
    _reject_total_redaction(asset, plan)
    written = writer.write(apply_plan(source, plan), frame_rate=plan.source_frame_rate)
    verified_sha256 = _verify_written_output(asset, written)

    attestation = DeidAttestation(
        media_id=asset.id,
        episode_id=asset.episode_id,
        media_kind=asset.kind,
        performed_at=clock(),
        performed_by=performed_by,
        policy=policy,
        plan=plan,
        source_sha256=asset.sha256,
        output_sha256=verified_sha256,
        output_uri=written.uri,
        output_frame_count=written.frame_count,
    )
    updated = asset.model_copy(
        update={
            "raw_uri": written.uri,
            "sha256": verified_sha256,
            "duration_seconds": written.frame_count / written.frame_rate,
            "frame_rate": written.frame_rate,
            "deid_status": DeidStatus.ATTESTED,
            "deid_attestation_sha256": attestation.digest,
        }
    )
    _record(trail, actor, AuditAction.DEID_ATTESTED, asset.episode_id, attestation.summary())
    return updated, attestation


def discard(
    asset: MediaAsset,
    policy: DeidPolicy,
    *,
    reason: str,
    performed_by: str,
    trail: AuditTrail | None = None,
    actor: Actor | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> tuple[MediaAsset, DeidAttestation]:
    """Destroy an asset rather than redact it, and record why.

    The default disposition for intraoperative audio (PLAN.md section 8).
    Discarding is a completed policy decision, so the resulting asset does not
    block its episode -- but it still produces an attestation, because "we
    destroyed it" is a claim that needs evidence as much as "we cleaned it".
    """
    _reject_settled(asset)
    attestation = DeidAttestation(
        media_id=asset.id,
        episode_id=asset.episode_id,
        media_kind=asset.kind,
        performed_at=clock(),
        performed_by=performed_by,
        policy=policy,
        plan=RedactionPlan(
            policy_version=policy.version,
            detectors=(),
            source_frame_count=0,
            source_frame_rate=1.0,
            analysis_stride_frames=policy.analysis_stride_frames,
        ),
        source_sha256=asset.sha256,
        output_sha256=None,
        output_uri=None,
        output_frame_count=None,
        discarded=True,
        discard_reason=reason,
    )
    updated = asset.model_copy(update={"deid_status": DeidStatus.DISCARDED})
    _record(
        trail,
        actor,
        AuditAction.DEID_DISCARDED,
        asset.episode_id,
        attestation.summary() | {"reason": reason},
    )
    return updated, attestation


def default_disposition(asset: MediaAsset, policy: DeidPolicy) -> str | None:
    """Reason this asset should be discarded under ``policy``, if it should.

    Returns:
        A human-readable reason, or ``None`` if the asset should be redacted
        rather than destroyed.
    """
    if asset.kind is MediaKind.AUDIO and policy.audio is AudioDisposition.DISCARD:
        return "intraoperative audio destroyed under the default policy (PLAN.md section 8)"
    if asset.kind is MediaKind.ROOM_VIDEO and policy.discard_room_video:
        return "room-facing video destroyed under the default policy (PLAN.md section 8)"
    return None


def _reject_settled(asset: MediaAsset) -> None:
    """Refuse to re-process media that has already reached a terminal state."""
    if asset.deid_status in (DeidStatus.ATTESTED, DeidStatus.DISCARDED):
        msg = (
            f"media {asset.id} is already {asset.deid_status.value}; "
            f"re-running de-identification would replace a settled attestation"
        )
        raise DeidentificationBoundaryError(msg)


def _record(
    trail: AuditTrail | None,
    actor: Actor | None,
    action: AuditAction,
    subject_ref: str,
    payload: dict[str, object],
) -> None:
    """Append to the trail when one was supplied."""
    if trail is None:
        return
    if actor is None:
        msg = "an audit trail was supplied without an actor; every entry needs a principal"
        raise ValueError(msg)
    trail.append(actor=actor, action=action, subject_ref=subject_ref, payload=payload)


def _require_analysed(asset: MediaAsset) -> None:
    """Refuse to redact media that has not been analysed.

    Enforces the ``analyze`` then ``redact`` ordering. Without it, ``redact``
    accepts a ``RAW`` asset and produces an attestation with no corresponding
    ``deid.started`` entry, so the audit trail shows a clearing with no
    preceding analysis -- and a caller holding a pre-analysis snapshot of an
    asset could mint a second attestation for the same media.
    """
    if asset.deid_status is not DeidStatus.IN_PROGRESS:
        msg = (
            f"media {asset.id} is {asset.deid_status.value}; redaction requires "
            f"an analysed asset, so call analyze() first and redact the asset "
            f"it returns"
        )
        raise DeidentificationBoundaryError(msg)


def _require_plan_matches_source(
    asset: MediaAsset, plan: RedactionPlan, source: FrameSource
) -> None:
    """Refuse to apply a plan built from different material.

    Without this check a plan analysed on one recording can be applied to
    another. The attack is cheap and total: analyse a clean three-frame clip,
    get a no-op plan, then apply it to the real recording and attest the
    result. Binding the plan to the frame count and rate it was built from
    makes the mismatch a hard error.
    """
    if plan.source_frame_count != source.frame_count:
        msg = (
            f"plan for media {asset.id} was built from {plan.source_frame_count} "
            f"frames but the source has {source.frame_count}; a plan may only be "
            f"applied to the material it was analysed on"
        )
        raise DeidentificationBoundaryError(msg)
    if plan.source_frame_rate != source.frame_rate:
        msg = (
            f"plan for media {asset.id} was built at {plan.source_frame_rate}fps "
            f"but the source is {source.frame_rate}fps; timestamps in the plan "
            f"would not line up with the frames"
        )
        raise DeidentificationBoundaryError(msg)


def _reject_total_redaction(asset: MediaAsset, plan: RedactionPlan) -> None:
    """Route a wholly out-of-body recording to ``discard`` rather than failing.

    A plan that drops every frame is a real outcome -- the camera never entered
    the patient -- but it is not a redaction. Letting it reach the writer
    surfaces as a bare "produced no frames" error with no indication of what to
    do instead.
    """
    if plan.drops_everything:
        msg = (
            f"the plan for media {asset.id} drops every frame, so there is no "
            f"recording left to attest; a wholly out-of-body capture should be "
            f"destroyed with discard() rather than redacted"
        )
        raise DeidentificationBoundaryError(msg)


def _verify_written_output(asset: MediaAsset, written: WrittenOutput) -> str:
    """Independently hash the writer's output and confirm its claim.

    ``FrameWriter`` is a protocol, so the digest it reports is an assertion by
    whatever implements it. Trusting that assertion reintroduces exactly the
    hole this pipeline exists to close: a writer that returns a fabricated
    digest yields an ``ATTESTED`` asset describing bytes nobody checked.

    Local outputs are therefore re-hashed from disk here. A writer targeting
    remote storage cannot be verified this way and is refused rather than
    trusted, because an unverifiable attestation is worse than none -- it looks
    like evidence.

    Returns:
        The digest computed here, not the one the writer reported.
    """
    parsed = urlparse(written.uri)
    if parsed.scheme != "file":
        msg = (
            f"writer for media {asset.id} produced a {parsed.scheme!r} output; "
            f"the alpha can only independently verify file:// outputs, and an "
            f"unverified digest must not become an attestation"
        )
        raise DeidentificationBoundaryError(msg)
    path = Path(url2pathname(unquote(parsed.path)))
    if not path.is_file():
        msg = f"writer for media {asset.id} reported {written.uri} but no file is there"
        raise DeidentificationBoundaryError(msg)
    actual = digest_file(path)
    if actual != written.sha256:
        msg = (
            f"writer for media {asset.id} reported digest {written.sha256} but "
            f"its output hashes to {actual}; the reported digest does not "
            f"describe the bytes on disk"
        )
        raise DeidentificationBoundaryError(msg)
    return actual
