"""Run a bound (task, agent) pair into a job directory.

P1: gym-policy (Lumen when installed; tests inject a factory).
P2: video-predict (labels vs JSON; AngioStress adds a claim footer).
P3: cartesian jobs and trajectory reconstitution live beside this module.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, assert_never

from or_audit.audit.canonical import digest
from or_audit.errors import TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.bind import assert_bind
from or_audit.eval.enums import AgentKind, PortId, WorldKind
from or_audit.eval.gym_world import GymFactory, make_gym, run_gym_episode, sample_action
from or_audit.eval.integrity import tree_digest
from or_audit.eval.job import (
    JobResult,
    TrialRecord,
    agent_identity,
    assemble_job_result,
    compute_head,
    read_job_config,
    read_job_result,
    resolve_bundle_path,
    write_job,
)
from or_audit.eval.plugins import load_policy_runtime, load_predictor_runtime
from or_audit.eval.predict import index_items, load_claim_footer, load_items
from or_audit.eval.reconstitute import assert_trajectory_matches_vector
from or_audit.eval.task import TaskSpec
from or_audit.eval.vector import project
from or_audit.eval.verifier import score_context

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
    task_package_digest = tree_digest(task_dir)
    agent_package_digest = (
        tree_digest(agent_dir) if agent_dir is not None else digest(agent.model_dump(mode="json"))
    )
    episodes = n if n is not None else task.environment.n_eval_episodes
    if episodes < 1:
        msg = f"n must be >= 1, got {episodes}"
        raise TaskContractError(msg)
    extra: dict[str, Any] = {}
    if task.port.id is PortId.GYM_POLICY:
        result, safety = _run_gym(
            task=task,
            task_dir=task_dir,
            agent=agent,
            agent_dir=agent_dir,
            task_digest=task_package_digest,
            agent_digest=agent_package_digest,
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
            task_digest=task_package_digest,
            agent_digest=agent_package_digest,
            n=episodes,
        )
    else:
        assert_never(task.port.id)
    config = {
        "task_id": task.id,
        "task_dir": "bundle/task",
        "agent_id": agent.id,
        "agent_dir": "bundle/agent" if agent_dir is not None else None,
        "task_digest": task_package_digest,
        "agent_digest": agent_package_digest,
        "n": result.n,
        "world_pin": task.environment.world_pin,
        "port": task.port.id.value,
        **extra,
    }
    write_job(
        out,
        config=config,
        result=result,
        task_dir=task_dir,
        agent_dir=agent_dir,
    )
    return result


def _run_gym(
    *,
    task: TaskSpec,
    task_dir: Path,
    agent: AgentPackage,
    agent_dir: Path | None,
    n: int,
    task_digest: str,
    agent_digest: str,
    gym_factory: GymFactory | None,
) -> tuple[JobResult, float]:
    if agent.kind not in {AgentKind.RANDOM, AgentKind.POLICY}:
        msg = f"gym-policy runner does not implement kind={agent.kind.value}"
        raise TaskContractError(msg)
    policy = None
    if agent.kind is AgentKind.POLICY:
        if agent_dir is None:
            raise TaskContractError(f"policy agent {agent.id} has no package directory")
        policy = load_policy_runtime(agent_dir, agent.entrypoint, agent.weights_path)

    factory: GymFactory = gym_factory or make_gym
    env = factory(task)
    identity = agent_identity(agent)
    unwrapped = getattr(env, "unwrapped", env)
    nested = getattr(unwrapped, "_env", unwrapped)
    safety = float(getattr(nested, "safety_max_pen", SAFETY_MAX_PEN))

    trials: list[TrialRecord] = []
    for seed in range(n):
        if policy is not None:
            policy.reset(seed=seed)

        def action_fn(world: Any, observation: Any, step: int, *, episode_seed: int = seed) -> Any:
            if policy is None:
                return sample_action(world, seed=episode_seed, step=step)
            return policy.act(observation, step=step)

        info, trajectory = run_gym_episode(env, seed=seed, action_fn=action_fn)
        vector = score_context(
            task=task,
            task_dir=task_dir,
            agent_identity=identity,
            seed=seed,
            context={
                "kind": "gym-policy",
                "info": info,
                "trajectory": trajectory,
                "safety_max_pen": safety,
            },
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
    return (
        assemble_job_result(
            task=task,
            agent=agent,
            trials=tuple(trials),
            task_digest=task_digest,
            agent_digest=agent_digest,
        ),
        safety,
    )


def _run_predict(
    *,
    task: TaskSpec,
    task_dir: Path,
    agent: AgentPackage,
    agent_dir: Path | None,
    n: int,
    task_digest: str,
    agent_digest: str,
) -> JobResult:
    inputs = load_items(task_dir / task.environment.inputs_path)
    labels = index_items(load_items(task_dir / task.environment.labels_path))
    if agent_dir is None:
        raise TaskContractError(f"agent {agent.id} has no package directory")
    predictor = load_predictor_runtime(agent_dir, agent.entrypoint, agent.weights_path)
    identity = agent_identity(agent)
    items = inputs[:n]
    trials: list[TrialRecord] = []
    for seed, item in enumerate(items):
        item_id = str(item["id"])
        if item_id not in labels:
            raise TaskContractError(f"task {task.id} has no label for item {item_id!r}")
        prediction = predictor.predict(item)
        if not isinstance(prediction, dict):
            raise TaskContractError(f"agent {agent.id} prediction for {item_id!r} is not an object")
        label = labels[item_id]
        context = {
            "kind": "video-predict",
            "input": item,
            "label": label,
            "prediction": prediction,
        }
        vector = score_context(
            task=task,
            task_dir=task_dir,
            agent_identity=identity,
            seed=seed,
            context=context,
        )
        trials.append(
            TrialRecord(
                seed=seed,
                vector=vector,
                trajectory=(context,),
            )
        )
    footer = ""
    if task.environment.kind is WorldKind.ANGIOSTRESS_CONTRACT:
        footer = load_claim_footer(task_dir / task.environment.contract_path)
    return assemble_job_result(
        task=task,
        agent=agent,
        trials=tuple(trials),
        task_digest=task_digest,
        agent_digest=agent_digest,
        claim_footer=footer,
    )


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
    task_dir = resolve_bundle_path(out, config["task_dir"], label="task")
    if tree_digest(task_dir) != config.get("task_digest"):
        raise TaskContractError("bundled task digest does not match config")
    task = load_task(task_dir)
    agent_dir_raw = config.get("agent_dir")
    if agent_dir_raw:
        agent_dir = resolve_bundle_path(out, agent_dir_raw, label="agent")
        if tree_digest(agent_dir) != config.get("agent_digest"):
            raise TaskContractError("bundled agent digest does not match config")
        agent = load_agent(agent_dir)
    else:
        agent_dir = None
        agent = builtin_random_agent()
        if digest(agent.model_dump(mode="json")) != config.get("agent_digest"):
            raise TaskContractError("builtin agent digest does not match config")
    assert_trajectory_matches_vector(
        out,
        task=task,
        task_dir=task_dir,
        result=previous,
        config=config,
    )
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
