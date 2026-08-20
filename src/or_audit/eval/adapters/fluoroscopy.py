"""Fluoroscopy and Angiography Modality Adapter (Endovascular Interventions).

Handles 2D pulsed fluoroscopy, Digital Subtraction Angiography (DSA), C-arm projective
geometry, catheter/guidewire action spaces, and interventional safety telemetry
(vessel wall penetration, arterial dissection risk, radiation area-dose product).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from or_audit.eval.adapters.base import ModalityAdapter
from or_audit.eval.enums import ModalityKind


@dataclass(frozen=True)
class FluoroscopyObservation:
    """Observation payload for image-guided endovascular interventions."""

    frame_index: int
    projection_frame_uri: str = ""
    dsa_contrast_active: bool = False
    carm_angles: tuple[float, float] = (0.0, 0.0)  # (LAO/RAO, CRAN/CAUD)
    roadmap_active: bool = False
    target_distance_mm: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CatheterGuidewireAction:
    """Action payload for robotic catheter/guidewire manipulation."""

    insertion_step_mm: float = 0.0
    rotation_step_deg: float = 0.0
    microcatheter_advance_mm: float = 0.0
    balloon_inflation_psi: float = 0.0
    contrast_inject_ml: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class FluoroscopyAdapter(ModalityAdapter):
    """Adapter for fluoroscopy and endovascular navigation."""

    modality: ModalityKind | str = ModalityKind.FLUOROSCOPY_DSA

    def __init__(self, modality: ModalityKind | str = ModalityKind.FLUOROSCOPY_DSA) -> None:
        self.modality = modality

    def validate_observation(self, observation: Any) -> bool:
        """Validate observation has X-ray projection or vascular state."""
        if isinstance(observation, FluoroscopyObservation):
            return observation.frame_index >= 0
        if isinstance(observation, dict):
            return (
                "projection_frame_uri" in observation
                or "carm_angles" in observation
                or "frame_index" in observation
                or "obs" in observation
                or "image" in observation
            )
        return hasattr(observation, "__array__") or isinstance(observation, (list, tuple))

    def validate_action(self, action: Any) -> bool:
        """Validate action contains insertion, rotation, or discrete command."""
        if isinstance(action, CatheterGuidewireAction):
            return True
        if isinstance(action, dict):
            return (
                "insertion_step_mm" in action
                or "rotation_step_deg" in action
                or "action" in action
                or "insertion_twist" in action
            )
        return isinstance(action, (int, float, str, list, tuple)) or hasattr(action, "__array__")

    def preprocess_observation(self, observation: Any) -> Any:
        if isinstance(observation, dict) and "frame_index" in observation:
            carm = observation.get("carm_angles")
            carm_tuple = tuple(carm) if isinstance(carm, (list, tuple)) else (0.0, 0.0)
            dist = observation.get("target_distance_mm")
            dist_val = float(dist) if dist is not None else None
            extra_dict = observation.get("extra")
            idx = observation.get("frame_index")
            return FluoroscopyObservation(
                frame_index=int(idx) if idx is not None else 0,
                projection_frame_uri=str(observation.get("projection_frame_uri") or ""),
                dsa_contrast_active=bool(observation.get("dsa_contrast_active", False)),
                carm_angles=carm_tuple,
                roadmap_active=bool(observation.get("roadmap_active", False)),
                target_distance_mm=dist_val,
                extra=extra_dict if isinstance(extra_dict, dict) else {},
            )
        return observation

    def postprocess_action(self, action: Any) -> Any:
        if isinstance(action, dict) and (
            "insertion_step_mm" in action or "rotation_step_deg" in action
        ):
            extra_dict = action.get("extra")
            return CatheterGuidewireAction(
                insertion_step_mm=float(action.get("insertion_step_mm") or 0.0),
                rotation_step_deg=float(action.get("rotation_step_deg") or 0.0),
                microcatheter_advance_mm=float(action.get("microcatheter_advance_mm") or 0.0),
                balloon_inflation_psi=float(action.get("balloon_inflation_psi") or 0.0),
                contrast_inject_ml=float(action.get("contrast_inject_ml") or 0.0),
                extra=extra_dict if isinstance(extra_dict, dict) else {},
            )
        return action

    def extract_safety_state(self, step_context: dict[str, Any] | None) -> dict[str, Any]:
        """Extract endovascular and fluoroscopic safety telemetry."""
        safety = super().extract_safety_state(step_context)
        if not isinstance(step_context, dict):
            return safety
        info = step_context.get("info")
        if isinstance(info, dict):
            for key in (
                "max_pen",
                "wall_force_n",
                "radiation_dose_mgy",
                "contrast_injected_ml",
                "dissection_risk",
                "spasm_detected",
                "thrombosis_risk",
            ):
                if key in info:
                    safety.setdefault(key, info[key])
        return safety

    def get_schema_spec(self) -> dict[str, Any]:
        """Return fluoroscopy modality schema metadata."""
        spec = super().get_schema_spec()
        spec.update(
            {
                "observation_type": "FluoroscopyObservation",
                "action_type": "CatheterGuidewireAction",
                "carm_tracking": True,
            }
        )
        return spec
