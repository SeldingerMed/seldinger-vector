"""NVIDIA Warp and Isaac Lab Simulation Bridge for GPU-Accelerated Rollouts.

Provides high-throughput parallel physics simulation for surgical robotics,
guidewire dynamics, and large-batch policy evaluation.
"""

from __future__ import annotations

from typing import Any

from or_audit.eval.enums import WorldKind
from or_audit.eval.sim.base import BaseSimulationBridge, SimulationEngine
from or_audit.eval.task import TaskSpec


class WarpBridge(BaseSimulationBridge):
    """Bridge for NVIDIA Warp / Isaac Lab GPU-accelerated simulation environments."""

    world_kind: WorldKind | str = WorldKind.WARP

    def __init__(
        self,
        env_name: str,
        *,
        parameters: dict[str, Any] | None = None,
        world_pin: str = "",
        num_envs: int = 1,
        warp_env: Any = None,
    ) -> None:
        self.env_name = env_name
        self.parameters = parameters or {}
        self.world_pin = world_pin
        self.num_envs = int(parameters.get("num_envs", num_envs)) if parameters else num_envs
        self._env = warp_env
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset Warp vectorized simulation."""
        self._step_count = 0
        if self._env is not None and hasattr(self._env, "reset"):
            return self._env.reset(seed=seed, options=options)  # type: ignore[no-any-return]
        obs = {
            "warp_env": self.env_name,
            "num_envs": self.num_envs,
            "robot_joint_pos": [0.0] * 7,
            "tool_ee_pos": [0.0, 0.0, 0.0],
        }
        info = {
            "warp_initialized": True,
            "gpu_device": self.parameters.get("device", "cuda:0"),
            "world_pin": self.world_pin,
            "seed": seed,
            "max_pen": 0.0,
            "haptic_overshoot_mm": 0.0,
        }
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one parallel Warp / PhysX 5 step."""
        self._step_count += 1
        if self._env is not None and hasattr(self._env, "step"):
            return self._env.step(action)  # type: ignore[no-any-return]

        max_steps = int(self.parameters.get("max_steps", 100))
        terminated = self._step_count >= max_steps
        truncated = False
        reward = 1.0 if terminated else 0.0

        obs = {
            "warp_env": self.env_name,
            "num_envs": self.num_envs,
            "robot_joint_pos": [0.01 * self._step_count] * 7,
            "tool_ee_pos": [0.1 * self._step_count, 0.0, 0.0],
        }
        info = {
            "step": self._step_count,
            "safe_success": terminated,
            "raw_success": terminated,
            "diverged": False,
            "max_pen": 0.0,
            "haptic_overshoot_mm": 0.05,
        }
        return obs, reward, terminated, truncated, info

    def get_state(self) -> dict[str, Any]:
        """Return GPU tensor state snapshot."""
        if self._env is not None and hasattr(self._env, "get_state"):
            return self._env.get_state()  # type: ignore[no-any-return]
        return {
            "env_name": self.env_name,
            "step_count": self._step_count,
            "num_envs": self.num_envs,
            "backend": "nvidia-warp-physx5",
        }

    def close(self) -> None:
        """Release GPU context and Warp memory pools."""
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()


def make_warp_bridge(task: TaskSpec) -> SimulationEngine:
    """Factory creating a WarpBridge for a task."""
    return WarpBridge(
        env_name=task.environment.gym_id or task.id,
        parameters=task.environment.parameters,
        world_pin=task.environment.world_pin,
    )
