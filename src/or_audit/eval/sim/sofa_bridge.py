"""SOFA Framework and SofaGym Simulation Bridge for Soft-Tissue and Biomechanics.

Provides biomechanical simulation for catheter Cosserat rods, vascular elasticity,
and soft-tissue deformation with graceful mock fallbacks for headless CI.
"""

from __future__ import annotations

from typing import Any

from or_audit.eval.enums import WorldKind
from or_audit.eval.sim.base import BaseSimulationBridge, SimulationEngine
from or_audit.eval.task import TaskSpec


class SofaBridge(BaseSimulationBridge):
    """Bridge for SOFA Framework / SofaGym biomechanical simulations."""

    world_kind: WorldKind | str = WorldKind.SOFA

    def __init__(
        self,
        scene_name: str,
        *,
        parameters: dict[str, Any] | None = None,
        world_pin: str = "",
        sofa_env: Any = None,
    ) -> None:
        self.scene_name = scene_name
        self.parameters = parameters or {}
        self.world_pin = world_pin
        self._env = sofa_env
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset SOFA simulation scene."""
        self._step_count = 0
        if self._env is not None and hasattr(self._env, "reset"):
            return self._env.reset(seed=seed, options=options)  # type: ignore[no-any-return]
        # Synthetic / fallback SOFA state
        obs = {
            "catheter_tip_xyz": (0.0, 0.0, 0.0),
            "beam_elements": 20,
            "tissue_stress_kpa": 0.1,
            "scene": self.scene_name,
        }
        info = {
            "sofa_initialized": True,
            "scene_name": self.scene_name,
            "world_pin": self.world_pin,
            "seed": seed,
            "max_pen": 0.0,
            "tissue_deformation_energy": 0.0,
        }
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one SOFA FEA / BeamFEM physics step."""
        self._step_count += 1
        if self._env is not None and hasattr(self._env, "step"):
            return self._env.step(action)  # type: ignore[no-any-return]

        # Synthetic biomechanical transition
        insertion = 0.0
        if isinstance(action, (int, float)):
            insertion = float(action)
        elif isinstance(action, dict):
            raw_ins = action.get("insertion_step_mm")
            if raw_ins is None:
                raw_ins = action.get("insertion")
            insertion = float(raw_ins) if raw_ins is not None else 1.0
        max_steps = int(self.parameters.get("max_steps", 100))
        terminated = self._step_count >= max_steps
        truncated = False
        reward = 1.0 if terminated else 0.0

        obs = {
            "catheter_tip_xyz": (0.0, 0.0, float(self._step_count * 1.5)),
            "beam_elements": 20,
            "tissue_stress_kpa": 0.2 + (0.05 * self._step_count),
        }
        info = {
            "step": self._step_count,
            "max_pen": 0.01 * (self._step_count / max(max_steps, 1)),
            "wall_force_n": 0.1 * insertion,
            "tissue_deformation_energy": 0.05 * self._step_count,
            "safe_success": terminated,
            "raw_success": terminated,
            "diverged": False,
        }
        return obs, reward, terminated, truncated, info

    def get_state(self) -> dict[str, Any]:
        """Return snapshot of underlying SOFA FEA nodes and tissue stresses."""
        if self._env is not None and hasattr(self._env, "get_state"):
            return self._env.get_state()  # type: ignore[no-any-return]
        return {
            "scene": self.scene_name,
            "step_count": self._step_count,
            "fem_mesh_nodes": 1024,
            "beam_elements": 20,
        }

    def close(self) -> None:
        """Release SOFA simulation context."""
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()


def make_sofa_bridge(task: TaskSpec) -> SimulationEngine:
    """Factory creating a SofaBridge for a task."""
    return SofaBridge(
        scene_name=task.environment.gym_id or task.id,
        parameters=task.environment.parameters,
        world_pin=task.environment.world_pin,
    )
