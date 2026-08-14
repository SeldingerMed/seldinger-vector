"""Gymnasium-shaped worlds for gym-policy tasks.

Lumen is optional. CI injects a factory. A published row still requires a
world pin; Newton is not a default dependency.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any, Protocol, cast

import numpy as np

from or_audit.errors import TaskContractError
from or_audit.eval.enums import WorldKind
from or_audit.eval.task import TaskSpec

GymFactory = Callable[[TaskSpec], "GymEnv"]


class GymEnv(Protocol):
    """Reset/step world. Lumen's NavEnv matches this without subclassing Gymnasium."""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Start an episode."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """One transition: obs, reward, terminated, truncated, info."""


def make_gym(task: TaskSpec) -> GymEnv:
    """Build the world named by ``task.environment``.

    Raises:
        TaskContractError: If Lumen/gymnasium is missing or the gym id is unknown.
    """
    kind = task.environment.kind
    if kind is WorldKind.LUMEN_GYM:
        return _make_lumen(
            task.environment.gym_id,
            world_pin=task.environment.world_pin,
            parameters=task.environment.parameters,
        )
    if kind is WorldKind.GYM:
        return _make_gymnasium(task.environment.gym_id, parameters=task.environment.parameters)
    msg = f"task {task.id} world kind {kind.value} is not a gym-policy world"
    raise TaskContractError(msg)


def _make_lumen(
    gym_id: str,
    *,
    world_pin: str,
    parameters: dict[str, bool | int | float | str],
) -> GymEnv:
    try:
        from lumen.envs.registration import LUMEN_ENVS, register_gym_envs
    except ImportError as exc:
        msg = (
            "this task requires Lumen. Install seldinger-lumen and Newton to "
            "run gym-policy evals (BUILD.md P1). Default CI does not import them."
        )
        raise TaskContractError(msg) from exc
    _require_lumen_pin(world_pin)
    if gym_id not in LUMEN_ENVS:
        known = ", ".join(sorted(LUMEN_ENVS))
        msg = f"unknown Lumen gym_id {gym_id!r}; known: {known}"
        raise TaskContractError(msg)
    try:
        import gymnasium

        register_gym_envs()
        kwargs: dict[str, Any] = dict(parameters)
        env = cast(GymEnv, gymnasium.make(gym_id, **kwargs))
    except ImportError:
        factory: Callable[..., Any] = LUMEN_ENVS[gym_id]
        env = cast(GymEnv, factory(**parameters))
    return env


def _require_lumen_pin(expected: str) -> None:
    """Require the installed VCS package to match the task's immutable world pin."""
    try:
        metadata = distribution("seldinger-lumen").read_text("direct_url.json")
    except PackageNotFoundError as exc:
        raise TaskContractError("seldinger-lumen distribution metadata is missing") from exc
    if not metadata:
        raise TaskContractError("seldinger-lumen is not installed from a pinned VCS revision")
    try:
        actual = json.loads(metadata)["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise TaskContractError("seldinger-lumen direct_url.json has no VCS commit") from exc
    if actual != expected:
        raise TaskContractError(f"Lumen pin mismatch: task requires {expected}, installed {actual}")


def _make_gymnasium(gym_id: str, *, parameters: dict[str, bool | int | float | str]) -> GymEnv:
    try:
        import gymnasium
    except ImportError as exc:
        msg = (
            f"task names gym {gym_id!r} but gymnasium is not installed; "
            f"pip install 'or-audit[lumen]' (gymnasium only) or the env's extra"
        )
        raise TaskContractError(msg) from exc
    kwargs: dict[str, Any] = dict(parameters)
    return cast(GymEnv, gymnasium.make(gym_id, **kwargs))


def sample_action(env: GymEnv, *, seed: int, step: int) -> Any:
    """Deterministic random action for a ``kind=random`` agent."""
    rng = np.random.default_rng(seed * 1_000_003 + step)
    space = getattr(env, "action_space", None)
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is not None and high is not None:
        return rng.uniform(np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64))
    return rng.uniform(-1.0, 1.0, size=2)


def jsonable(value: Any) -> Any:
    """Convert numpy values so a trajectory can be JSON and canonical."""
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and not np.isfinite(item):
            return 0.0
        return item
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return 0.0
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def run_gym_episode(
    env: GymEnv,
    *,
    seed: int,
    action_fn: Callable[[GymEnv, Any, int], Any],
    max_steps: int = 10_000,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Roll out one episode. Returns final info and the trajectory."""
    obs, _reset_info = env.reset(seed=seed)
    steps: list[dict[str, Any]] = []
    info: Any = {}
    for step_i in range(max_steps):
        action = action_fn(env, obs, step_i)
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append(
            {
                "action": jsonable(action),
                "obs": jsonable(obs),
                "reward": jsonable(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": jsonable(info) if isinstance(info, dict) else {},
            }
        )
        if terminated or truncated:
            break
    if not isinstance(info, dict):
        info = {}
    return info, tuple(steps)
