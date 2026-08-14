"""Validate task-owned verifier output into a non-collapsible trial vector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from or_audit.domain.enums import GateStatus
from or_audit.errors import TaskContractError
from or_audit.eval.plugins import load_verifier_runtime
from or_audit.eval.task import TaskSpec
from or_audit.eval.vector import GateOutcome, MetricOutcome, TrialVector


def score_context(
    *,
    task: TaskSpec,
    task_dir: Path,
    agent_identity: str,
    seed: int,
    context: dict[str, Any],
) -> TrialVector:
    """Run the task package's verifier and require its declared vector shape."""
    if not task.verifier.entrypoint:
        raise TaskContractError(f"task {task.id} has no verifier entrypoint")
    raw = load_verifier_runtime(task_dir, task.verifier.entrypoint).score(context)
    if not isinstance(raw, dict):
        raise TaskContractError(f"task {task.id} verifier must return an object")
    raw_gates = raw.get("gates")
    raw_metrics = raw.get("metrics")
    if not isinstance(raw_gates, dict) or not isinstance(raw_metrics, dict):
        raise TaskContractError("verifier output requires gates and metrics objects")

    declared_gates = [gate.id for gate in task.verifier.gates]
    declared_metrics = [metric.id for metric in task.verifier.metrics]
    if set(raw_gates) != set(declared_gates):
        raise TaskContractError(
            f"verifier gates {sorted(raw_gates)} do not match "
            f"declared gates {sorted(declared_gates)}"
        )
    if set(raw_metrics) != set(declared_metrics):
        raise TaskContractError(
            "verifier metrics "
            f"{sorted(raw_metrics)} do not match declared metrics {sorted(declared_metrics)}"
        )

    gates: list[GateOutcome] = []
    for gate_id in declared_gates:
        outcome = raw_gates[gate_id]
        if not isinstance(outcome, dict):
            raise TaskContractError(f"gate {gate_id} outcome must be an object")
        status_raw = outcome.get("status")
        if not isinstance(status_raw, str):
            raise TaskContractError(f"gate {gate_id} has invalid status")
        try:
            status = GateStatus(status_raw)
        except ValueError as exc:
            raise TaskContractError(f"gate {gate_id} has invalid status") from exc
        reason = outcome.get("reason", "")
        if not isinstance(reason, str):
            raise TaskContractError(f"gate {gate_id} reason must be text")
        gates.append(GateOutcome(id=gate_id, status=status, reason=reason))

    metrics: list[MetricOutcome] = []
    for metric_id in declared_metrics:
        value = raw_metrics[metric_id]
        if value is not None and not isinstance(value, bool | int | float):
            raise TaskContractError(f"metric {metric_id} must be boolean, numeric, or null")
        metrics.append(
            MetricOutcome(
                id=metric_id,
                value=value,
                headline=metric_id == task.verifier.headline,
            )
        )
    return TrialVector(
        task_id=task.id,
        task_version=task.task_version,
        agent_identity=agent_identity,
        seed=seed,
        gates=tuple(gates),
        metrics=tuple(metrics),
    )
