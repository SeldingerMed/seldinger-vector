"""Simulation bridges and physics connectors for procedural evaluation."""

from __future__ import annotations

from or_audit.eval.sim.base import (
    BaseSimulationBridge,
    SimFactory,
    SimulationEngine,
    clear_simulation_registry,
    get_simulation_engine,
    list_simulation_engines,
    register_simulation_engine,
    require_simulation_engine,
    reset_default_simulation_engines,
)
from or_audit.eval.sim.gym_bridge import GymnasiumBridge, make_gym_bridge
from or_audit.eval.sim.sofa_bridge import SofaBridge, make_sofa_bridge
from or_audit.eval.sim.warp_bridge import WarpBridge, make_warp_bridge

reset_default_simulation_engines()

__all__ = [
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
    "register_simulation_engine",
    "require_simulation_engine",
    "reset_default_simulation_engines",
]
