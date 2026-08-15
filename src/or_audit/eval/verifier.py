"""Run task-owned verifiers separately and validate typed vector output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from or_audit.domain.enums import GateStatus
from or_audit.errors import TaskContractError
from or_audit.eval.plugins import VerifierRuntime, load_verifier_runtime
from or_audit.eval.task import TaskSpec
from or_audit.eval.vector import GateOutcome, MetricOutcome, TrialVector


def score_context(
    *,
    task: TaskSpec,
    task_dir: Path,
    agent_identity: str,
    seed: int,
    context: dict[str, Any],
    runtime: VerifierRuntime | None = None,
) -> TrialVector:
    """Score oracle evidence in a verifier process that is separate from the agent."""
    if not task.verifier.entrypoint:
        raise TaskContractError(f"task {task.id} has no verifier entrypoint")
    verifier = runtime or load_verifier_runtime(task_dir, task.verifier.entrypoint)
    owns_runtime = runtime is None
    try:
        raw = verifier.score(context)
    finally:
        if owns_runtime:
            close = getattr(verifier, "close", None)
            if callable(close):
                close()
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
            f"verifier gates {sorted(raw_gates)} do not match declared gates "
            f"{sorted(declared_gates)}"
        )
    if set(raw_metrics) != set(declared_metrics):
        raise TaskContractError(
            f"verifier metrics {sorted(raw_metrics)} do not match declared metrics "
            f"{sorted(declared_metrics)}"
        )

    gates = []
    for gate_id in declared_gates:
        outcome = raw_gates[gate_id]
        if not isinstance(outcome, dict):
            raise TaskContractError(f"gate {gate_id} outcome must be an object")
        raw_status = outcome.get("status")
        if not isinstance(raw_status, str):
            raise TaskContractError(f"gate {gate_id} has invalid status")
        try:
            status = GateStatus(raw_status)
        except ValueError as exc:
            raise TaskContractError(f"gate {gate_id} has invalid status") from exc
        reason = outcome.get("reason", "")
        if not isinstance(reason, str):
            raise TaskContractError(f"gate {gate_id} reason must be text")
        gates.append(GateOutcome(id=gate_id, status=status, reason=reason))

    metrics = []
    for metric_id in declared_metrics:
        definition = task.metric(metric_id)
        value = raw_metrics[metric_id]
        if value is not None and not isinstance(value, bool | int | float | str):
            raise TaskContractError(f"metric {metric_id} returned an unsupported value")
        if (
            definition.kind.value == "categorical"
            and value is not None
            and value not in definition.categories
        ):
            raise TaskContractError(
                f"categorical metric {metric_id} returned undeclared category {value!r}"
            )
        metrics.append(
            MetricOutcome(
                id=metric_id,
                value=value,
                kind=definition.kind,
                unit=definition.unit,
                direction=definition.direction,
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
