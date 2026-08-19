"""SOFA Framework and SofaGym Simulation Bridge for Soft-Tissue and Biomechanics.

Provides biomechanical simulation for catheter Cosserat rods, vascular elasticity,
and soft-tissue deformation. A synthetic stand-in exists for headless CI, but it is
refused unless the task opts in, and it is stamped into every artifact it touches.
"""

from __future__ import annotations

from typing import Any

from or_audit.errors import TaskContractError
from or_audit.eval.enums import WorldKind
from or_audit.eval.sim.base import (
    BACKEND_REAL,
    BACKEND_SYNTHETIC_STUB,
    BaseSimulationBridge,
    SimulationEngine,
    module_distribution_version,
    world_kind_key,
)
from or_audit.eval.task import TaskSpec

_SOFA_MODULES = "'Sofa' / 'SofaRuntime'"


def _refuse_synthetic_sofa(kind: str) -> str:
    return (
        f"world kind {kind!r} has no SOFA backend attached: the SOFA python bindings "
        f"({_SOFA_MODULES}) did not yield a runnable scene. A synthetic stand-in would "
        "invent tissue stress, wall force, and penetration numbers that this task's hard "
        "safety gates would then score as physical evidence. Install SOFA, or set "
        "environment.synthetic_stub = true in task.toml to accept a non-physical "
        'stand-in (artifacts are stamped backend="synthetic-stub" and export-rl refuses '
        "the run)."
    )


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
        allow_synthetic: bool = False,
        backend_version: str = "",
    ) -> None:
        if sofa_env is None and not allow_synthetic:
            raise TaskContractError(_refuse_synthetic_sofa(world_kind_key(self.world_kind)))
        self.scene_name = scene_name
        self.parameters = parameters or {}
        self.world_pin = world_pin
        self._env = sofa_env
        self._backend_version = backend_version
        self._step_count = 0

    def engine_provenance(self) -> dict[str, Any]:
        """Report whether a real SOFA scene or the synthetic stand-in produced the data."""
        synthetic = self._env is None
        return {
            "engine": world_kind_key(self.world_kind),
            "backend": BACKEND_SYNTHETIC_STUB if synthetic else BACKEND_REAL,
            "backend_version": "" if synthetic else self._backend_version,
            "world_pin": self.world_pin,
        }

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
            "backend": BACKEND_SYNTHETIC_STUB,
        }
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one SOFA FEA / BeamFEM physics step."""
        self._step_count += 1
        if self._env is not None and hasattr(self._env, "step"):
            return self._env.step(action)  # type: ignore[no-any-return]

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
            "backend": BACKEND_SYNTHETIC_STUB,
        }
        return obs, reward, terminated, truncated, info

    def get_state(self) -> dict[str, Any]:
        """Return snapshot of underlying SOFA FEA nodes and tissue stresses."""
        if self._env is not None and hasattr(self._env, "get_state"):
            return self._env.get_state()  # type: ignore[no-any-return]
        return {
            "scene": self.scene_name,
            "step_count": self._step_count,
            "backend": BACKEND_SYNTHETIC_STUB,
        }

    def close(self) -> None:
        """Release SOFA simulation context."""
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()


def _acquire_sofa_env(task: TaskSpec) -> tuple[Any, str]:
    """Best-effort acquisition of a real SOFA-backed scene: returns (env, version)."""
    try:
        import Sofa
        import SofaRuntime
    except ImportError:
        return None, ""
    detected = module_distribution_version("Sofa")
    if not detected:
        detected = str(
            getattr(Sofa, "__version__", "") or getattr(SofaRuntime, "__version__", "") or ""
        )
    scene_id = task.environment.gym_id
    if not scene_id:
        return None, detected
    try:
        import gymnasium
        import sofagym  # noqa: F401
    except ImportError:
        return None, detected
    kwargs: dict[str, Any] = dict(task.environment.parameters)
    try:
        return gymnasium.make(scene_id, **kwargs), detected
    except Exception:  # SofaGym raises engine-specific errors for an unknown scene
        return None, detected


def make_sofa_bridge(task: TaskSpec) -> SimulationEngine:
    """Factory creating a SofaBridge for a task, preferring a real SOFA scene."""
    sofa_env, backend_version = _acquire_sofa_env(task)
    return SofaBridge(
        scene_name=task.environment.gym_id or task.id,
        parameters=dict(task.environment.parameters),
        world_pin=task.environment.world_pin,
        sofa_env=sofa_env,
        allow_synthetic=task.environment.synthetic_stub,
        backend_version=backend_version,
    )
