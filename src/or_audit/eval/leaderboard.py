"""Static, vector-preserving leaderboard generation."""

from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from or_audit.errors import TaskContractError
from or_audit.eval.integrity import tree_digest
from or_audit.eval.job import (
    JobResult,
    compute_head,
    read_job_config,
    read_job_result,
    resolve_bundle_path,
)
from or_audit.eval.loader import load_task
from or_audit.eval.reconstitute import assert_trajectory_matches_vector


def _result_paths(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        if (path / "result.json").is_file() and (path / "config.json").is_file():
            found.add(path)
        elif path.is_dir():
            found.update(
                candidate.parent
                for candidate in path.rglob("result.json")
                if (candidate.parent / "config.json").is_file()
            )
    if not found:
        raise TaskContractError("no job result.json files found")
    return sorted(found)


def _metric_summary(result: JobResult) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for trial in result.trials:
        for metric in trial.vector.metrics:
            if metric.value is not None:
                values.setdefault(metric.id, []).append(float(metric.value))
    return {metric_id: fmean(samples) for metric_id, samples in sorted(values.items())}


def _verified_result(job_dir: Path) -> JobResult:
    config = read_job_config(job_dir)
    result = read_job_result(job_dir)
    if compute_head(result) != result.head:
        raise TaskContractError(f"result head mismatch in {job_dir}")
    if config.get("task_digest") != result.task_digest:
        raise TaskContractError(f"task digest mismatch in {job_dir}")
    if config.get("agent_digest") != result.agent_digest:
        raise TaskContractError(f"agent digest mismatch in {job_dir}")
    task_dir = resolve_bundle_path(job_dir, config["task_dir"], label="task")
    if tree_digest(task_dir) != result.task_digest:
        raise TaskContractError(f"bundled task digest mismatch in {job_dir}")
    agent_dir_raw = config.get("agent_dir")
    if (
        agent_dir_raw
        and tree_digest(resolve_bundle_path(job_dir, agent_dir_raw, label="agent"))
        != result.agent_digest
    ):
        raise TaskContractError(f"bundled agent digest mismatch in {job_dir}")
    task = load_task(task_dir)
    assert_trajectory_matches_vector(
        job_dir,
        task=task,
        task_dir=task_dir,
        result=result,
        config=config,
    )
    return result


def leaderboard_data(paths: list[Path]) -> dict[str, Any]:
    """Load verified jobs and return deterministic task-scoped rows."""
    rows: list[dict[str, Any]] = []
    for job_dir in _result_paths(paths):
        result = _verified_result(job_dir)
        assessed = result.headline_true + result.headline_false
        rows.append(
            {
                "task_id": result.task_id,
                "task_version": result.task_version,
                "agent_identity": result.agent_identity,
                "world_pin": result.world_pin,
                "n": result.n,
                "headline": result.headline,
                "headline_true": result.headline_true,
                "headline_false": result.headline_false,
                "headline_unassessable": result.headline_unassessable,
                "headline_rate": result.headline_true / assessed if assessed else None,
                "any_gate_failed": result.any_gate_failed,
                "metrics": _metric_summary(result),
                "head": result.head,
            }
        )
    rows.sort(
        key=lambda row: (
            row["task_id"],
            -(row["headline_rate"] if row["headline_rate"] is not None else -1.0),
            row["any_gate_failed"],
            row["agent_identity"],
        )
    )
    return {"format_version": "1", "rows": rows}


def render_html(data: dict[str, Any]) -> str:
    """Render a dependency-free static table without collapsing the vector."""
    body: list[str] = []
    for row in data["rows"]:
        rate = "—" if row["headline_rate"] is None else f"{100 * row['headline_rate']:.1f}%"
        metrics = ", ".join(f"{key}={value:.4g}" for key, value in row["metrics"].items())
        cells = (
            row["task_id"],
            row["agent_identity"],
            row["world_pin"] or "—",
            str(row["n"]),
            f"{row['headline']} {rate}",
            str(row["headline_unassessable"]),
            str(row["any_gate_failed"]),
            metrics,
            row["head"],
        )
        body.append(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>"
        )
    return f"""<!doctype html>
<meta charset="utf-8">
<title>OR-Audit safety-vector leaderboard</title>
<style>
body{{font:15px system-ui,sans-serif;margin:2rem;color:#17202a}}
h1{{margin-bottom:.25rem}}
p{{color:#52606d}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d8dee4;padding:.55rem;text-align:left;vertical-align:top}}
th{{background:#f4f6f8}}
td:last-child{{font:12px ui-monospace,monospace;overflow-wrap:anywhere}}
tr:nth-child(even){{background:#fafbfc}}
</style>
<h1>OR-Audit safety-vector leaderboard</h1>
<p>Ranked within each task by headline rate. Safety gates, abstentions, metrics, pins,
and artifact heads remain visible; there is no cross-task overall score.</p>
<table><thead><tr>
<th>Task</th><th>Agent</th><th>World pin</th><th>n</th><th>Headline</th>
<th>Unassessable</th><th>Gate failures</th><th>Metrics</th><th>Artifact head</th>
</tr></thead><tbody>{"".join(body)}</tbody></table>
"""


def write_leaderboard(paths: list[Path], out: Path) -> dict[str, Any]:
    """Write deterministic ``leaderboard.json`` and ``index.html``."""
    data = leaderboard_data(paths)
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (out / "index.html").write_text(render_html(data), encoding="utf-8")
    return data
