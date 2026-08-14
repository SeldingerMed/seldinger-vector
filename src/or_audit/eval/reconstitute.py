"""Reconstitute a trial vector from its trajectory without stepping the world."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.job import JobResult
from or_audit.eval.task import TaskSpec
from or_audit.eval.vector import TrialVector
from or_audit.eval.verifier import score_context

DEFAULT_SAFETY_MAX_PEN = 0.3


def reconstitute_trial_vector(
    trial_dir: Path,
    *,
    task: TaskSpec,
    task_dir: Path,
    agent_identity: str,
    seed: int,
    safety_max_pen: float = DEFAULT_SAFETY_MAX_PEN,
) -> TrialVector:
    """Map ``trajectory.json`` back through the task-owned verifier."""
    path = trial_dir / "trajectory.json"
    if not path.is_file():
        msg = f"missing trajectory.json in {trial_dir}"
        raise TaskContractError(msg)
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        msg = f"{trial_dir.name} trajectory must be a non-empty JSON array"
        raise TaskContractError(msg)
    first = raw[0]
    if not isinstance(first, dict):
        msg = f"{trial_dir.name} trajectory items must be objects"
        raise TaskContractError(msg)
    if "info" in first and "action" in first:
        last = raw[-1]
        if not isinstance(last, dict):
            msg = f"{trial_dir.name} last gym step is not an object"
            raise TaskContractError(msg)
        info = last.get("info")
        if not isinstance(info, dict):
            msg = f"{trial_dir.name} last gym step is missing info"
            raise TaskContractError(msg)
        context = {
            "kind": "gym-policy",
            "info": info,
            "trajectory": raw,
            "safety_max_pen": safety_max_pen,
        }
    elif first.get("kind") == "video-predict":
        if len(raw) != 1:
            msg = f"{trial_dir.name} video-predict trajectory must have exactly one item"
            raise TaskContractError(msg)
        context = first
    else:
        msg = f"{trial_dir.name} trajectory is neither gym-policy (action+info) nor video-predict"
        raise TaskContractError(msg)
    return score_context(
        task=task,
        task_dir=task_dir,
        agent_identity=agent_identity,
        seed=seed,
        context=context,
    )


def assert_trajectory_matches_vector(
    job_dir: Path,
    *,
    task: TaskSpec,
    task_dir: Path,
    result: JobResult,
    config: dict[str, Any],
) -> None:
    """Refuse a job whose stored trajectory does not reconstitute its vector."""
    raw_pen = config.get("safety_max_pen", DEFAULT_SAFETY_MAX_PEN)
    if isinstance(raw_pen, bool) or not isinstance(raw_pen, int | float):
        msg = f"{job_dir} config safety_max_pen must be numeric"
        raise TaskContractError(msg)
    safety = float(raw_pen)
    for trial in result.trials:
        trial_dir = job_dir / f"trial-{result.task_id}-{trial.seed}"
        recon = reconstitute_trial_vector(
            trial_dir,
            task_dir=task_dir,
            task=task,
            agent_identity=result.agent_identity,
            seed=trial.seed,
            safety_max_pen=safety,
        )
        if recon != trial.vector:
            msg = (
                f"{trial_dir.name}: trajectory reconstitutes a different vector "
                f"than result.json; a published trial must replay from its trajectory"
            )
            raise ScoreContractError(msg)
