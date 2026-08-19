"""Robotic Kinematics and Orthopedic Modality Adapter.

Handles robot manipulator joint states, end-effector SE(3) poses, haptic safety boundaries,
orthopedic bone milling/burring telemetry, and collision/force safety limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from or_audit.eval.adapters.base import ModalityAdapter, register_adapter
from or_audit.eval.enums import ModalityKind


@dataclass(frozen=True)
class KinematicObservation:
    """Observation payload for robotic manipulators and orthopedic robotics."""

    joint_positions: tuple[float, ...] = ()
    joint_velocities: tuple[float, ...] = ()
    ee_position_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_orientation_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)
    haptic_boundary_distance_mm: float | None = None
    measured_force_n: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KinematicAction:
    """Action payload for robotic manipulation and orthopedic cutting."""

    joint_velocity_cmd: tuple[float, ...] = ()
    target_ee_position_xyz: tuple[float, float, float] | None = None
    target_ee_orientation_quat: tuple[float, float, float, float] | None = None
    cutting_rpm: float = 0.0
    gripper_state: float = 0.0  # 0.0 (closed) to 1.0 (fully open)
    extra: dict[str, Any] = field(default_factory=dict)


class KinematicsAdapter(ModalityAdapter):
    """Adapter for robotic kinematics and orthopedic robotics."""

    modality: ModalityKind | str = ModalityKind.ROBOTIC_KINEMATICS

    def __init__(self, modality: ModalityKind | str = ModalityKind.ROBOTIC_KINEMATICS) -> None:
        self.modality = modality

    def validate_observation(self, observation: Any) -> bool:
        """Validate observation has joint states or end-effector telemetry."""
        if isinstance(observation, KinematicObservation):
            return True
        if isinstance(observation, dict):
            return (
                "joint_positions" in observation
                or "ee_position_xyz" in observation
                or "state" in observation
                or "obs" in observation
            )
        return hasattr(observation, "__array__") or isinstance(observation, (list, tuple))

    def validate_action(self, action: Any) -> bool:
        """Validate action contains joint or Cartesian commands."""
        if isinstance(action, KinematicAction):
            return True
        if isinstance(action, dict):
            return (
                "joint_velocity_cmd" in action
                or "target_ee_position_xyz" in action
                or "action" in action
            )
        return isinstance(action, (int, float, str, list, tuple)) or hasattr(action, "__array__")

    def preprocess_observation(self, observation: Any) -> Any:
        """Normalize raw dictionary into KinematicObservation."""
        if isinstance(observation, dict) and (
            "joint_positions" in observation or "ee_position_xyz" in observation
        ):
            raw_joints = observation.get("joint_positions")
            raw_vels = observation.get("joint_velocities")
            raw_pos = observation.get("ee_position_xyz")
            raw_quat = observation.get("ee_orientation_quat")
            joints = tuple(raw_joints) if isinstance(raw_joints, (list, tuple)) else ()
            vels = tuple(raw_vels) if isinstance(raw_vels, (list, tuple)) else ()
            pos = (
                tuple(raw_pos)
                if isinstance(raw_pos, (list, tuple)) and len(raw_pos) == 3
                else (0.0, 0.0, 0.0)
            )
            quat = (
                tuple(raw_quat)
                if isinstance(raw_quat, (list, tuple)) and len(raw_quat) == 4
                else (0.0, 0.0, 0.0, 1.0)
            )
            haptic_dist = observation.get("haptic_boundary_distance_mm")
            force = observation.get("measured_force_n")
            extra_dict = observation.get("extra")
            return KinematicObservation(
                joint_positions=joints,
                joint_velocities=vels,
                ee_position_xyz=pos,
                ee_orientation_quat=quat,
                haptic_boundary_distance_mm=float(haptic_dist) if haptic_dist is not None else None,
                measured_force_n=float(force) if force is not None else None,
                extra=extra_dict if isinstance(extra_dict, dict) else {},
            )
        return observation

    def postprocess_action(self, action: Any) -> Any:
        """Normalize action dictionary into KinematicAction."""
        if isinstance(action, dict) and (
            "joint_velocity_cmd" in action or "target_ee_position_xyz" in action
        ):
            vel_cmd = tuple(action.get("joint_velocity_cmd", ()))
            pos_cmd = (
                tuple(action["target_ee_position_xyz"])
                if "target_ee_position_xyz" in action
                else None
            )
            quat_cmd = (
                tuple(action["target_ee_orientation_quat"])
                if "target_ee_orientation_quat" in action
                else None
            )
            extra_dict = action.get("extra")
            return KinematicAction(
                joint_velocity_cmd=vel_cmd,
                target_ee_position_xyz=pos_cmd,
                target_ee_orientation_quat=quat_cmd,
                cutting_rpm=float(action.get("cutting_rpm") or 0.0),
                gripper_state=float(action.get("gripper_state") or 0.0),
                extra=extra_dict if isinstance(extra_dict, dict) else {},
            )
        return action

    def extract_safety_state(self, step_context: dict[str, Any] | None) -> dict[str, Any]:
        """Extract kinematic and orthopedic safety telemetry."""
        safety = super().extract_safety_state(step_context)
        if not isinstance(step_context, dict):
            return safety
        info = step_context.get("info")
        if isinstance(info, dict):
            for key in (
                "haptic_boundary_overshoot_mm",
                "joint_limit_margin_deg",
                "excessive_traction_force_n",
                "collision_detected",
                "bone_burr_temperature_c",
            ):
                if key in info:
                    safety.setdefault(key, info[key])
        return safety

    def get_schema_spec(self) -> dict[str, Any]:
        """Return kinematics modality schema metadata."""
        spec = super().get_schema_spec()
        spec.update(
            {
                "observation_type": "KinematicObservation",
                "action_type": "KinematicAction",
                "pose_representation": "SE3_Quat",
            }
        )
        return spec


register_adapter(ModalityKind.ROBOTIC_KINEMATICS, KinematicsAdapter, override=True)
register_adapter(
    ModalityKind.ORTHOPEDIC_POINTCLOUD,
    lambda **kw: KinematicsAdapter(ModalityKind.ORTHOPEDIC_POINTCLOUD, **kw),
    override=True,
)
