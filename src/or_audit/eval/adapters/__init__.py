"""Modality adapters for procedural healthcare AI evaluation."""

from __future__ import annotations

from or_audit.eval.adapters.base import (
    BaseModalityAdapter,
    ModalityAdapter,
    adapter_revision,
    clear_registry,
    get_adapter,
    list_adapters,
    register_adapter,
    require_adapter,
    reset_default_adapters,
)

# Verify plugin pins and register adapters from the manifest BEFORE importing
# any concrete adapter module, so no plugin module is executed before its
# binding digest is checked against the manifest.
reset_default_adapters()

from or_audit.eval.adapters.endoluminal import (  # noqa: E402 -- after bootstrap
    EndoluminalAction,
    EndoluminalAdapter,
    EndoluminalObservation,
)
from or_audit.eval.adapters.fluoroscopy import (  # noqa: E402 -- after bootstrap
    CatheterGuidewireAction,
    FluoroscopyAdapter,
    FluoroscopyObservation,
)
from or_audit.eval.adapters.kinematics import (  # noqa: E402 -- after bootstrap
    KinematicAction,
    KinematicObservation,
    KinematicsAdapter,
)
from or_audit.eval.adapters.video import (  # noqa: E402 -- after bootstrap
    VideoAdapter,
    VideoFrameObservation,
    VideoToolAction,
)

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
    "adapter_revision",
    "clear_registry",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "require_adapter",
    "reset_default_adapters",
]
