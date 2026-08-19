"""Modality adapters for procedural healthcare AI evaluation."""

from __future__ import annotations

from or_audit.eval.adapters.base import (
    BaseModalityAdapter,
    ModalityAdapter,
    clear_registry,
    get_adapter,
    list_adapters,
    register_adapter,
    require_adapter,
    reset_default_adapters,
)
from or_audit.eval.adapters.endoluminal import (
    EndoluminalAction,
    EndoluminalAdapter,
    EndoluminalObservation,
)
from or_audit.eval.adapters.fluoroscopy import (
    CatheterGuidewireAction,
    FluoroscopyAdapter,
    FluoroscopyObservation,
)
from or_audit.eval.adapters.kinematics import (
    KinematicAction,
    KinematicObservation,
    KinematicsAdapter,
)
from or_audit.eval.adapters.video import (
    VideoAdapter,
    VideoFrameObservation,
    VideoToolAction,
)

reset_default_adapters()

__all__ = [
    "BaseModalityAdapter",
    "CatheterGuidewireAction",
    "EndoluminalAction",
    "EndoluminalAdapter",
    "EndoluminalObservation",
    "FluoroscopyAdapter",
    "FluoroscopyObservation",
    "KinematicAction",
    "KinematicObservation",
    "KinematicsAdapter",
    "ModalityAdapter",
    "VideoAdapter",
    "VideoFrameObservation",
    "VideoToolAction",
    "clear_registry",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "require_adapter",
    "reset_default_adapters",
]
