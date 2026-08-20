"""Simulation bridges and physics connectors for procedural evaluation."""

from __future__ import annotations

from or_audit.eval.sim.base import (
    BACKEND_REAL,
    BACKEND_SYNTHETIC_STUB,
    BACKEND_UNKNOWN,
    BaseSimulationBridge,
    SimFactory,
    SimulationEngine,
    clear_simulation_registry,
    get_simulation_engine,
    list_simulation_engines,
    module_distribution_version,
    register_simulation_engine,
    require_simulation_engine,
    reset_default_simulation_engines,
    world_kind_key,
)
from or_audit.eval.sim.gym_bridge import GymnasiumBridge, make_gym_bridge
from or_audit.eval.sim.sofa_bridge import SofaBridge, make_sofa_bridge
from or_audit.eval.sim.warp_bridge import WarpBridge, make_warp_bridge

reset_default_simulation_engines()

__all__ = [
    "BACKEND_REAL",
    "BACKEND_SYNTHETIC_STUB",
    "BACKEND_UNKNOWN",
    "BaseSimulationBridge",
    "GymnasiumBridge",
    "SimFactory",
    "SimulationEngine",
    "SofaBridge",
    "WarpBridge",
    "clear_simulation_registry",
    "get_simulation_engine",
    "list_simulation_engines",
    "make_gym_bridge",
    "make_sofa_bridge",
    "make_warp_bridge",
    "module_distribution_version",
    "register_simulation_engine",
    "require_simulation_engine",
    "reset_default_simulation_engines",
    "world_kind_key",
]
