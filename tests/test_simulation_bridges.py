"""Tests for simulation bridges (Gymnasium, SOFA Framework, NVIDIA Warp / Isaac Lab)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from or_audit.eval.enums import WorldKind
from or_audit.eval.loader import load_task
from or_audit.eval.runner import builtin_random_agent, run_job
from or_audit.eval.sim import (
    BaseSimulationBridge,
    GymnasiumBridge,
    SimulationEngine,
    SofaBridge,
    WarpBridge,
    clear_simulation_registry,
    list_simulation_engines,
    register_simulation_engine,
    reset_default_simulation_engines,
)


def test_simulation_engine_protocols() -> None:
    sofa = SofaBridge("TestScene")
    assert isinstance(sofa, SimulationEngine)
    assert isinstance(sofa, BaseSimulationBridge)

    warp = WarpBridge("TestWarp")
    assert isinstance(warp, SimulationEngine)
    assert isinstance(warp, BaseSimulationBridge)

    class MockEnv:
        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[Any, dict[str, Any]]:
            return {}, {}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            return {}, 0.0, False, False, {}

    gym_bridge = GymnasiumBridge(MockEnv())
    assert isinstance(gym_bridge, SimulationEngine)
    assert isinstance(gym_bridge, BaseSimulationBridge)
    assert gym_bridge.unwrapped is not None


def test_sofa_bridge_lifecycle() -> None:
    sofa = SofaBridge(
        scene_name="AneurysmCoiling",
        parameters={"max_steps": 10},
        world_pin="sofa-pin-v1",
    )
    assert sofa.world_kind == WorldKind.SOFA

    obs, info = sofa.reset(seed=42)
    assert "catheter_tip_xyz" in obs
    assert info["sofa_initialized"] is True
    assert info["scene_name"] == "AneurysmCoiling"

    # Step simulation with explicit zero insertion
    next_obs, _reward, term, _trunc, step_info = sofa.step({"insertion_step_mm": 0.0})
    assert "tissue_stress_kpa" in next_obs
    assert step_info["wall_force_n"] == 0.0
    assert not term
    state = sofa.get_state()
    assert state["fem_mesh_nodes"] == 1024
    assert state["scene"] == "AneurysmCoiling"

    sofa.close()


def test_warp_bridge_lifecycle() -> None:
    warp = WarpBridge(
        env_name="SurgicalSuture-v0",
        parameters={"max_steps": 10, "num_envs": 16, "device": "cuda:0"},
        world_pin="warp-pin-v1",
    )
    assert warp.world_kind == WorldKind.WARP
    assert warp.num_envs == 16

    obs, info = warp.reset(seed=123)
    assert obs["num_envs"] == 16
    assert info["warp_initialized"] is True

    # Step
    next_obs, _reward, _term, _trunc, step_info = warp.step([0.0] * 7)
    assert "robot_joint_pos" in next_obs
    assert step_info["haptic_overshoot_mm"] == 0.05
    state = warp.get_state()
    assert state["backend"] == "nvidia-warp-physx5"

    warp.close()


def test_simulation_registry() -> None:
    reset_default_simulation_engines()
    engines = list_simulation_engines()
    assert "sofa" in engines
    assert "warp" in engines
    assert "isaac-lab" in engines
    assert "gym" in engines

    # Custom factory
    class MockCustomSim(BaseSimulationBridge):
        world_kind = "custom-sim"

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[Any, dict[str, Any]]:
            return {"custom": True}, {"seed": seed}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            return {"custom": True}, 1.0, True, False, {"safe_success": True}

    register_simulation_engine("custom-sim", lambda task: MockCustomSim())
    assert "custom-sim" in list_simulation_engines()

    clear_simulation_registry()
    assert len(list_simulation_engines()) == 0
    reset_default_simulation_engines()


def test_run_closed_loop_with_simulation_bridges(tmp_path: Path) -> None:
    reset_default_simulation_engines()
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Test Broncho Airway Nav with Random Agent
    broncho_task_dir = repo_root / "docs/examples/tasks/broncho-airway-nav"
    broncho_task = load_task(broncho_task_dir)
    random_agent = builtin_random_agent("broncho-steering")
    out_dir = tmp_path / "broncho-run"

    # Mock gym environment for broncho
    class MockBronchoGym:
        def __init__(self) -> None:
            self._step = 0

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[Any, dict[str, Any]]:
            self._step = 0
            return {"airway_id": "RB1"}, {"seed": seed}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            self._step += 1
            term = self._step >= 5
            return (
                {"airway_id": "RB1_sub"},
                1.0 if term else 0.0,
                term,
                False,
                {
                    "target_reached": term,
                    "max_contact_force_n": 0.5,
                    "safe_navigation": term,
                    "diverged": False,
                },
            )

    res = run_job(
        task=broncho_task,
        task_dir=broncho_task_dir,
        agent=random_agent,
        agent_dir=None,
        out=out_dir,
        n=2,
        gym_factory=lambda task: MockBronchoGym(),
    )

    assert res.n == 2
    assert len(res.trials) == 2
    assert res.trials[0].vector.gates[0].status == "pass"
    m_safe = res.trials[0].vector.metric("safe_navigation")
    assert m_safe is not None
    assert m_safe.value is True
