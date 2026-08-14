"""video-predict scoring: labels the task author brought vs agent JSON.

The kernel does not know CABG from cath. Field names come from the task.
AngioStress is this adapter with a claim footer and a contract JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from or_audit.domain.enums import GateStatus
from or_audit.errors import TaskContractError
from or_audit.eval.task import TaskSpec
from or_audit.eval.vector import GateOutcome, MetricOutcome, TrialVector


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.is_file():
        msg = f"missing {path.name}: {path}"
        raise TaskContractError(msg)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"{path} must be a JSON object"
        raise TaskContractError(msg)
    return data


def load_items(path: Path) -> tuple[dict[str, Any], ...]:
    """Load ``{"items": [...]}`` from a labels or predictions file."""
    data = load_json_object(path)
    raw = data.get("items")
    if not isinstance(raw, list) or not raw:
        msg = f"{path} must contain a non-empty items array"
        raise TaskContractError(msg)
    items: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict) or "id" not in entry:
            msg = f"{path} items must be objects with an id"
            raise TaskContractError(msg)
        items.append(entry)
    return tuple(items)


def load_claim_footer(contract_path: Path) -> str:
    """Copy the contract's claim boundary. Empty is invalid for AngioStress."""
    data = load_json_object(contract_path)
    footer = data.get("claim_boundary")
    if not isinstance(footer, str) or not footer.strip():
        msg = (
            f"{contract_path} is missing claim_boundary; an AngioStress-shaped "
            f"result without a claim footer is not a scorecard"
        )
        raise TaskContractError(msg)
    return footer.strip()


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value == value


def vector_from_prediction(
    *,
    task: TaskSpec,
    agent_identity: str,
    seed: int,
    label: dict[str, Any],
    prediction: dict[str, Any],
) -> TrialVector:
    """Score one labelled item. Metric ids are the task's, not a medical ontology."""
    abstained = bool(prediction.get("abstain", False))
    gates: list[GateOutcome] = []
    for gate in task.verifier.gates:
        if gate.id == "unsafe_prediction":
            if abstained and task.verifier.abstain_ok:
                gates.append(
                    GateOutcome(
                        id=gate.id,
                        status=GateStatus.NOT_ASSESSABLE,
                        reason="agent abstained",
                    )
                )
            elif bool(prediction.get("unsafe", False)):
                gates.append(
                    GateOutcome(
                        id=gate.id,
                        status=GateStatus.FAIL,
                        reason="prediction marked unsafe",
                    )
                )
            else:
                gates.append(
                    GateOutcome(
                        id=gate.id,
                        status=GateStatus.PASS,
                        reason="prediction not marked unsafe",
                    )
                )
        elif gate.id == "finite_metric_check":
            dice = prediction.get("dice")
            finite = _finite_number(dice)
            if "finite_metric_check" in prediction:
                finite = finite and bool(prediction["finite_metric_check"])
            gates.append(
                GateOutcome(
                    id=gate.id,
                    status=GateStatus.PASS if finite else GateStatus.FAIL,
                    reason="required predict metrics finite" if finite else "non-finite metric",
                )
            )
        else:
            gates.append(
                GateOutcome(
                    id=gate.id,
                    status=GateStatus.NOT_ASSESSABLE,
                    reason=f"no scorer for gate {gate.id}",
                )
            )

    metrics: list[MetricOutcome] = []
    for metric in task.verifier.metrics:
        headline = metric.id == task.verifier.headline
        if metric.id == "abstained":
            metrics.append(MetricOutcome(id=metric.id, value=abstained, headline=headline))
            continue
        if abstained and task.verifier.abstain_ok and metric.id != "abstained":
            metrics.append(MetricOutcome(id=metric.id, value=None, headline=headline))
            continue
        if metric.id in {"next_step_correct", "outcome_correct"}:
            field = "next_step" if metric.id == "next_step_correct" else "outcome"
            ok = label.get(field) == prediction.get(field)
            metrics.append(MetricOutcome(id=metric.id, value=ok, headline=headline))
            continue
        if metric.id == "contract_validation_passed":
            dice_ok = _finite_number(prediction.get("dice"))
            passed = dice_ok
            if "contract_validation_passed" in prediction:
                passed = bool(prediction["contract_validation_passed"]) and dice_ok
            metrics.append(MetricOutcome(id=metric.id, value=passed, headline=headline))
            continue
        if metric.id in prediction and _finite_number(prediction[metric.id]):
            metrics.append(
                MetricOutcome(id=metric.id, value=float(prediction[metric.id]), headline=headline)
            )
            continue
        raw = prediction.get(metric.id, label.get(metric.id))
        if isinstance(raw, bool):
            metrics.append(MetricOutcome(id=metric.id, value=raw, headline=headline))
        elif isinstance(raw, int | float) and not isinstance(raw, bool):
            metrics.append(MetricOutcome(id=metric.id, value=float(raw), headline=headline))
        else:
            metrics.append(MetricOutcome(id=metric.id, value=_as_bool(raw), headline=headline))

    return TrialVector(
        task_id=task.id,
        task_version=task.task_version,
        agent_identity=agent_identity,
        seed=seed,
        gates=tuple(gates),
        metrics=tuple(metrics),
    )


def index_items(items: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    """Map item id -> object. Duplicate ids are a contract error."""
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item["id"])
        if item_id in out:
            msg = f"duplicate item id {item_id!r}"
            raise TaskContractError(msg)
        out[item_id] = item
    return out
