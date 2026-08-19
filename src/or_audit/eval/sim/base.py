"""Universal Simulation Engine Protocol and Registry for Physical Healthcare AI.

Bridges diverse physics engines (Gymnasium, Lumen, SOFA Framework, NVIDIA Warp/Isaac Lab,
PyBullet) into a unified reset/step/render/inspect interface for procedural evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from or_audit.errors import TaskContractError
from or_audit.eval.enums import WorldKind
from or_audit.eval.task import TaskSpec


@runtime_checkable
class SimulationEngine(Protocol):
    """Protocol for physics simulators and procedural worlds."""

    world_kind: WorldKind | str

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Start an episode: returns (initial_observation, info_dict)."""
        ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one physics step: returns (obs, reward, terminated, truncated, info)."""
        ...

    def render(self, mode: str = "rgb_array") -> Any:
        """Render the current simulation state (image frame, point cloud, or mesh)."""
        ...

    def close(self) -> None:
        """Release simulator resources and subprocess/GPU handles."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Return snapshot of underlying physics state for oracle/verifier evaluation."""
        ...


class BaseSimulationBridge:
    """Base class for simulation bridges providing default protocol behaviors."""

    world_kind: WorldKind | str = WorldKind.GYM

    def render(self, mode: str = "rgb_array") -> Any:
        """Default render implementation (returns None if headless)."""
        return None

    def close(self) -> None:
        """Default close implementation (no-op)."""

    def get_state(self) -> dict[str, Any]:
        """Default state extractor (returns empty dictionary)."""
        return {}


SimFactory = Callable[[TaskSpec], SimulationEngine]
_SIM_ENGINE_REGISTRY: dict[str, SimFactory] = {}


def register_simulation_engine(
    kind: WorldKind | str,
    factory: SimFactory,
    *,
    override: bool = False,
) -> None:
    """Register a simulation engine factory for a world kind."""
    key = kind.value if isinstance(kind, WorldKind) else str(kind)
    if key in _SIM_ENGINE_REGISTRY and not override:
        raise TaskContractError(f"simulation engine already registered for {key!r}")
    _SIM_ENGINE_REGISTRY[key] = factory


def get_simulation_engine(task: TaskSpec) -> SimulationEngine | None:
    """Get an instantiated simulation engine for a task, or None if not registered."""
    key = (
        task.environment.kind.value
        if isinstance(task.environment.kind, WorldKind)
        else str(task.environment.kind)
    )
    factory = _SIM_ENGINE_REGISTRY.get(key)
    if factory is None:
        return None
    return factory(task)


def require_simulation_engine(task: TaskSpec) -> SimulationEngine:
    """Get a simulation engine for a task or raise TaskContractError if missing."""
    engine = get_simulation_engine(task)
    if engine is None:
        key = (
            task.environment.kind.value
            if isinstance(task.environment.kind, WorldKind)
            else str(task.environment.kind)
        )
        known = ", ".join(sorted(_SIM_ENGINE_REGISTRY.keys()))
        raise TaskContractError(
            f"task {task.id} world kind {key!r} has no registered simulation engine; known: {known}"
        )
    return engine


def list_simulation_engines() -> dict[str, str]:
    """Return dictionary of registered world kinds and their factory names."""
    return {k: getattr(v, "__name__", "factory") for k, v in sorted(_SIM_ENGINE_REGISTRY.items())}


def clear_simulation_registry() -> None:
    """Reset the simulation registry (primarily for test isolation)."""
    _SIM_ENGINE_REGISTRY.clear()


def reset_default_simulation_engines() -> None:
    """Reset and re-register standard built-in simulation engine bridges."""
    _SIM_ENGINE_REGISTRY.clear()
    from or_audit.eval.sim.gym_bridge import make_gym_bridge
    from or_audit.eval.sim.sofa_bridge import make_sofa_bridge
    from or_audit.eval.sim.warp_bridge import make_warp_bridge

    register_simulation_engine(WorldKind.LUMEN_GYM, make_gym_bridge, override=True)
    register_simulation_engine(WorldKind.GYM, make_gym_bridge, override=True)
    register_simulation_engine(WorldKind.SOFA, make_sofa_bridge, override=True)
    register_simulation_engine(WorldKind.WARP, make_warp_bridge, override=True)
    register_simulation_engine(WorldKind.ISAAC_LAB, make_warp_bridge, override=True)
    register_simulation_engine(WorldKind.PYBULLET, make_gym_bridge, override=True)
