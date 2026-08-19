"""Endoluminal Modality Adapter (Robotic Bronchoscopy and Airway Navigation).

Handles endoscopic video with 3D CT airway trees, EM tracking coordinate transforms,
catheter steering kinematics (bend, roll, insertion), and airway safety telemetry
(wall contact force, puncture risk, off-target biopsy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from or_audit.eval.adapters.base import ModalityAdapter, register_adapter
from or_audit.eval.enums import ModalityKind


@dataclass(frozen=True)
class EndoluminalObservation:
    """Observation payload for endoluminal navigation (e.g. robotic bronchoscopy)."""

    frame_index: int
    camera_frame_uri: str = ""
    airway_id: str = ""
    em_sensor_pose: tuple[float, float, float, float, float, float] | None = None
    target_distance_mm: float | None = None
    branch_level: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EndoluminalAction:
    """Action payload for robotic bronchoscope or catheter steering."""

    bend_angle_deg: float = 0.0
    roll_angle_deg: float = 0.0
    insertion_mm: float = 0.0
    biopsy_deployed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class EndoluminalAdapter(ModalityAdapter):
    """Adapter for robotic bronchoscopy and endoluminal procedures."""

    modality: ModalityKind | str = ModalityKind.AIRWAY_BRONCHOSCOPY

    def validate_observation(self, observation: Any) -> bool:
        """Validate observation has airway or EM sensor data."""
        if isinstance(observation, EndoluminalObservation):
            return observation.frame_index >= 0
        if isinstance(observation, dict):
            return (
                "airway_id" in observation
                or "em_sensor_pose" in observation
                or "camera_frame" in observation
                or "frame_index" in observation
            )
        return hasattr(observation, "__array__") or isinstance(observation, (list, tuple))

    def validate_action(self, action: Any) -> bool:
        """Validate action contains steering commands or discrete step."""
        if isinstance(action, EndoluminalAction):
            return True
        if isinstance(action, dict):
            return (
                "bend_angle_deg" in action
                or "insertion_mm" in action
                or "action" in action
                or "steering" in action
            )
        return isinstance(action, (int, float, str, list, tuple)) or hasattr(action, "__array__")

    def preprocess_observation(self, observation: Any) -> Any:
        """Normalize raw dictionary into EndoluminalObservation."""
        if isinstance(observation, dict) and (
            "airway_id" in observation or "frame_index" in observation
        ):
            raw_pose = observation.get("em_sensor_pose")
            pose_tuple = tuple(raw_pose) if isinstance(raw_pose, (list, tuple)) else None
            dist = observation.get("target_distance_mm")
            dist_val = float(dist) if dist is not None else None
            extra_dict = observation.get("extra")
            idx = observation.get("frame_index")
            lvl = observation.get("branch_level")
            return EndoluminalObservation(
                frame_index=int(idx) if idx is not None else 0,
                camera_frame_uri=str(observation.get("camera_frame_uri") or ""),
                airway_id=str(observation.get("airway_id") or ""),
                em_sensor_pose=pose_tuple,
                target_distance_mm=dist_val,
                branch_level=int(lvl) if lvl is not None else 0,
                extra=extra_dict if isinstance(extra_dict, dict) else {},
            )
        return observation

    def postprocess_action(self, action: Any) -> Any:
        """Normalize action dictionary into EndoluminalAction."""
        if isinstance(action, dict) and ("bend_angle_deg" in action or "insertion_mm" in action):
            extra_dict = action.get("extra")
            bend = action.get("bend_angle_deg")
            roll = action.get("roll_angle_deg")
            ins = action.get("insertion_mm")
            return EndoluminalAction(
                bend_angle_deg=float(bend) if bend is not None else 0.0,
                roll_angle_deg=float(roll) if roll is not None else 0.0,
                insertion_mm=float(ins) if ins is not None else 0.0,
                biopsy_deployed=bool(action.get("biopsy_deployed", False)),
                extra=extra_dict if isinstance(extra_dict, dict) else {},
            )
        return action

    def extract_safety_state(self, step_context: dict[str, Any] | None) -> dict[str, Any]:
        """Extract bronchoscopic safety telemetry."""
        safety = super().extract_safety_state(step_context)
        if not isinstance(step_context, dict):
            return safety
        info = step_context.get("info")
        if isinstance(info, dict):
            for key in (
                "contact_force_n",
                "wall_pressure_kpa",
                "wall_puncture",
                "off_target_biopsy",
                "airway_obstruction",
            ):
                if key in info:
                    safety.setdefault(key, info[key])
        return safety

    def get_schema_spec(self) -> dict[str, Any]:
        """Return endoluminal modality schema metadata."""
        spec = super().get_schema_spec()
        spec.update(
            {
                "observation_type": "EndoluminalObservation",
                "action_type": "EndoluminalAction",
                "control_space": "bend_roll_insertion",
            }
        )
        return spec


register_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY, EndoluminalAdapter, override=True)
