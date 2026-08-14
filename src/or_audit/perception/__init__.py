"""Perception: the boundary between pixels and judgement."""

from __future__ import annotations

from or_audit.perception.backend import (
    ANNOTATION_BACKEND,
    AnnotationBackend,
    PerceptionBackend,
    episode_duration_s,
    readable_video,
)
from or_audit.perception.observations import (
    BLEEDING_RANK,
    BleedingEvent,
    BleedingSeverity,
    Confidence,
    CriticalStructure,
    CvsCriterion,
    CvsObservation,
    PerceptionResult,
    PhaseSegment,
    ProximityEvent,
    StructureSighting,
    SurgicalPhase,
)

__all__ = [
    "ANNOTATION_BACKEND",
    "BLEEDING_RANK",
    "AnnotationBackend",
    "BleedingEvent",
    "BleedingSeverity",
    "Confidence",
    "CriticalStructure",
    "CvsCriterion",
    "CvsObservation",
    "PerceptionBackend",
    "PerceptionResult",
    "PhaseSegment",
    "ProximityEvent",
    "StructureSighting",
    "SurgicalPhase",
    "episode_duration_s",
    "readable_video",
]
