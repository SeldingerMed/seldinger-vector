"""Run a bound (task, agent) pair into a job directory.

P1: gym-policy (Lumen when installed; tests inject a factory).
P2: video-predict (labels vs JSON; AngioStress adds a claim footer).
P3: cartesian jobs and trajectory reconstitution live beside this module.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, assert_never

from or_audit.errors import TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.bind import assert_bind
from or_audit.eval.enums import AgentKind, PortId, WorldKind
from or_audit.eval.gym_world import GymFactory, make_gym, run_gym_episode, sample_action
from or_audit.eval.job import (
    JobResult,
    TrialRecord,
    agent_identity,
    assemble_job_result,
    compute_head,
    read_job_config,
    read_job_result,
    write_job,
)
from or_audit.eval.predict import (
    index_items,
    load_claim_footer,
    load_items,
    vector_from_prediction,
)
from or_audit.eval.reconstitute import assert_trajectory_matches_vector
from or_audit.eval.task import TaskSpec
from or_audit.eval.vector import project, vector_from_lumen_info

SAFETY_MAX_PEN = 0.3


def builtin_random_agent() -> AgentPackage:
    """``-a random`` — gym-policy baseline, no weights."""
    return AgentPackage(
        format_version="1",
        id="seldingermed/random",
        agent_version="0",
        port=PortId.GYM_POLICY,
        kind=AgentKind.RANDOM,
    )


def run_job(
    *,
    task: TaskSpec,
    task_dir: Path,
    agent: AgentPackage,
    agent_dir: Path | None,
    out: Path,
    n: int | None = None,
    gym_factory: GymFactory | None = None,
) -> JobResult:
    """Bind, run, write a job directory, return the :class:`JobResult`."""
    assert_bind(task, agent)
    task.assert_runnable()
    episodes = n if n is not None else task.environment.n_eval_episodes
    if episodes < 1:
        msg = f"n must be >= 1, got {episodes}"
        raise TaskContractError(msg)
    extra: dict[str, Any] = {}
    if task.port.id is PortId.GYM_POLICY:
        result, safety = _run_gym(
            task=task,
            agent=agent,
            n=episodes,
            gym_factory=gym_factory,
        )
        extra["safety_max_pen"] = safety
    elif task.port.id is PortId.VIDEO_PREDICT:
        result = _run_predict(
            task=task,
            task_dir=task_dir,
            agent=agent,
            agent_dir=agent_dir,
            n=episodes,
        )
    else:
        assert_never(task.port.id)
    config = {
        "task_id": task.id,
        "task_dir": str(task_dir.resolve()),
        "agent_id": agent.id,
        "agent_dir": str(agent_dir.resolve()) if agent_dir is not None else None,
        "n": result.n,
        "world_pin": task.environment.world_pin,
        "port": task.port.id.value,
        **extra,
    }
    write_job(out, config=config, result=result)
    return result


def _run_gym(
    *,
    task: TaskSpec,
    agent: AgentPackage,
    n: int,
    gym_factory: GymFactory | None,
) -> tuple[JobResult, float]:
    if agent.kind is AgentKind.POLICY and not agent.entrypoint:
        msg = (
            f"agent {agent.id} is kind=policy with no entrypoint; P1 runs "
            f"kind=random (and a policy entrypoint later)"
        )
        raise TaskContractError(msg)
    if agent.kind not in {AgentKind.RANDOM, AgentKind.POLICY}:
        msg = f"gym-policy runner does not implement kind={agent.kind.value}"
        raise TaskContractError(msg)

    factory: GymFactory = gym_factory or make_gym
    env = factory(task)
    identity = agent_identity(agent)
    safety = float(getattr(env, "safety_max_pen", SAFETY_MAX_PEN))

    trials: list[TrialRecord] = []
    for seed in range(n):

        def action_fn(world: Any, step: int, *, episode_seed: int = seed) -> Any:
            if agent.kind is AgentKind.RANDOM:
                return sample_action(world, seed=episode_seed, step=step)
            msg = f"policy entrypoint {agent.entrypoint!r} is not loaded in P1"
            raise TaskContractError(msg)

        info, trajectory = run_gym_episode(env, seed=seed, action_fn=action_fn)
        vector = vector_from_lumen_info(
            task=task,
            agent_identity=identity,
            seed=seed,
            info=info,
            safety_max_pen=safety,
        )
        projection = project(vector, task.projection) if task.projection is not None else None
        trials.append(
            TrialRecord(
                seed=seed,
                vector=vector,
                trajectory=trajectory,
                projection=projection,
            )
        )
    return assemble_job_result(task=task, agent=agent, trials=tuple(trials)), safety


def _run_predict(
    *,
    task: TaskSpec,
    task_dir: Path,
    agent: AgentPackage,
    agent_dir: Path | None,
    n: int,
) -> JobResult:
    labels_path = task_dir / task.environment.labels_path
    labels = load_items(labels_path)
    if agent_dir is None or not agent.predictions_path:
        msg = (
            f"agent {agent.id} has no predictions_path; video-predict P2 scores "
            f"submitted JSON against labels (a live VLM entrypoint is P6)"
        )
        raise TaskContractError(msg)
    predictions = index_items(load_items(agent_dir / agent.predictions_path))
    identity = agent_identity(agent)
    items = labels[:n]
    trials: list[TrialRecord] = []
    for seed, label in enumerate(items):
        item_id = str(label["id"])
        if item_id not in predictions:
            msg = f"agent {agent.id} has no prediction for item {item_id!r}"
            raise TaskContractError(msg)
        vector = vector_from_prediction(
            task=task,
            agent_identity=identity,
            seed=seed,
            label=label,
            prediction=predictions[item_id],
        )
        trials.append(
            TrialRecord(
                seed=seed,
                vector=vector,
                trajectory=({"id": item_id, "label": label, "prediction": predictions[item_id]},),
            )
        )
    footer = ""
    if task.environment.kind is WorldKind.ANGIOSTRESS_CONTRACT:
        footer = load_claim_footer(task_dir / task.environment.contract_path)
    return assemble_job_result(task=task, agent=agent, trials=tuple(trials), claim_footer=footer)


def replay_job(
    out: Path,
    *,
    load_task: Callable[[Path], TaskSpec],
    load_agent: Callable[[Path], AgentPackage],
    gym_factory: GymFactory | None = None,
) -> JobResult:
    """Re-run a job from its config and check the head."""
    config = read_job_config(out)
    previous = read_job_result(out)
    task_dir = Path(str(config["task_dir"]))
    task = load_task(task_dir)
    agent_dir_raw = config.get("agent_dir")
    if agent_dir_raw:
        agent_dir = Path(str(agent_dir_raw))
        agent = load_agent(agent_dir)
    else:
        agent_dir = None
        agent = builtin_random_agent()
    assert_trajectory_matches_vector(out, task=task, result=previous, config=config)
    rerun = run_job(
        task=task,
        task_dir=task_dir,
        agent=agent,
        agent_dir=agent_dir,
        out=out,
        n=int(config["n"]),
        gym_factory=gym_factory,
    )
    if rerun.head != previous.head:
        msg = (
            f"replay head mismatch: stored {previous.head} reran {rerun.head}; "
            f"a published row must replay (BUILD.md §1.3)"
        )
        raise TaskContractError(msg)
    if compute_head(rerun) != rerun.head:
        msg = "rerun stamped a head that does not match its payload"
        raise TaskContractError(msg)
    return rerun
