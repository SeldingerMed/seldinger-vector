"""Export a job of vectors as jsonl of versioned projection floats for RL.

The leaderboard still reads the vector. Training may only see a closed
projection. Homemade floats are refused: ``--projection`` is a
:class:`~or_audit.eval.enums.ProjectionId`, and each row is recomputed from
the stored vector rather than trusted from ``projection.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.cartesian import iter_job_dirs
from or_audit.eval.enums import ProjectionId
from or_audit.eval.integrity import tree_digest
from or_audit.eval.job import read_job_config, read_job_result, resolve_bundle_path
from or_audit.eval.loader import load_task
from or_audit.eval.task import ProjectionSpec, TaskSpec
from or_audit.eval.vector import project


class RlExportRecord(BaseModel):
    """One episode in an RL dump. Not a leaderboard row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    task_version: str
    agent_identity: str
    world_pin: str
    seed: Annotated[int, Field(ge=0)]
    episode_id: str
    projection_id: str
    projection_version: str
    projection: float


def _spec_for_task(task: TaskSpec, projection_id: ProjectionId) -> ProjectionSpec:
    if task.projection is None:
        return ProjectionSpec(id=projection_id, version="0")
    return ProjectionSpec(id=projection_id, version=task.projection.version)


def export_job_records(job_dir: Path, *, projection_id: ProjectionId) -> tuple[RlExportRecord, ...]:
    """Recompute one job's trials into RL records."""
    config = read_job_config(job_dir)
    result = read_job_result(job_dir)
    task_dir = resolve_bundle_path(job_dir, config["task_dir"], label="task")
    if tree_digest(task_dir) != config.get("task_digest"):
        raise TaskContractError(f"{job_dir.name} bundled task digest does not match config")
    task = load_task(task_dir)
    spec = _spec_for_task(task, projection_id)
    records: list[RlExportRecord] = []
    for trial in result.trials:
        recomputed = project(trial.vector, spec)
        if trial.projection is not None and trial.projection != recomputed:
            msg = (
                f"{job_dir.name} seed {trial.seed}: stored projection "
                f"{trial.projection} disagrees with {spec.id.value}={recomputed}; "
                f"export-rl recomputes from the vector and will not ship a "
                f"homemade float"
            )
            raise ScoreContractError(msg)
        records.append(
            RlExportRecord(
                task_id=result.task_id,
                task_version=result.task_version,
                agent_identity=result.agent_identity,
                world_pin=result.world_pin,
                seed=trial.seed,
                episode_id=f"{result.task_id}-{trial.seed}",
                projection_id=spec.id.value,
                projection_version=spec.version,
                projection=recomputed,
            )
        )
    return tuple(records)


def export_rl(
    path: Path,
    *,
    projection_id: ProjectionId,
    out: Path,
) -> int:
    """Write jsonl for a job directory or a cartesian parent. Returns episode count."""
    records: list[RlExportRecord] = []
    for job_dir in iter_job_dirs(path):
        records.extend(export_job_records(job_dir, projection_id=projection_id))
    if not records:
        msg = f"{path} exported no episodes"
        raise TaskContractError(msg)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.model_dump(mode="json"), sort_keys=True) for record in records]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(records)
