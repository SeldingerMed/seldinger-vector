"""De-identification: detectors, policy, plans, and attestation.

PLAN.md section 8 treats this as a named work-stream rather than a pipeline
step, and the module is sized accordingly. The invariant it exists to hold:
media reaches ``ATTESTED`` only after bytes were written and hashed here.
"""

from __future__ import annotations

from or_audit.deid.attestation import DeidAttestation
from or_audit.deid.detectors import (
    PixelBox,
    TimeSegment,
    detect_out_of_body,
    detect_static_overlays,
    redness_ratio,
)
from or_audit.deid.pipeline import analyze, default_disposition, discard, redact
from or_audit.deid.plan import PlannedBox, PlannedSegment, RedactionPlan, apply_plan
from or_audit.deid.policy import SAFE_OVERLAY_MIN_PX, AudioDisposition, DeidPolicy
from or_audit.deid.writer import FrameWriter, NpzFrameWriter, WrittenOutput

__all__ = [
    "SAFE_OVERLAY_MIN_PX",
    "AudioDisposition",
    "DeidAttestation",
    "DeidPolicy",
    "FrameWriter",
    "NpzFrameWriter",
    "PixelBox",
    "PlannedBox",
    "PlannedSegment",
    "RedactionPlan",
    "TimeSegment",
    "WrittenOutput",
    "analyze",
    "apply_plan",
    "default_disposition",
    "detect_out_of_body",
    "detect_static_overlays",
    "discard",
    "redact",
    "redness_ratio",
]
