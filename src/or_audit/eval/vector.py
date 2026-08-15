"""Typed gate and metric vectors with declarative projection rules."""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from or_audit.domain.enums import GateStatus
from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.contracts import (
    GateProjectionPolicy,
    MetricDirection,
    MetricKind,
)
from or_audit.eval.task import ProjectionSpec


class GateOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    status: GateStatus
    reason: str = ""


class MetricOutcome(BaseModel):
    """One typed metric; ``None`` is explicitly unassessable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    value: bool | float | str | None
    kind: MetricKind | None = None
    unit: str = ""
    direction: MetricDirection = MetricDirection.NEUTRAL
    headline: bool = False

    @model_validator(mode="before")
    @classmethod
    def _infer_legacy_kind(cls, raw: Any) -> Any:
        if not isinstance(raw, dict) or raw.get("kind") is not None:
            return raw
        data = dict(raw)
        value = data.get("value")
        if isinstance(value, bool) or value is None:
            data["kind"] = MetricKind.BOOLEAN.value
        elif isinstance(value, str):
            data["kind"] = MetricKind.CATEGORICAL.value
        else:
            data["kind"] = MetricKind.CONTINUOUS.value
        return data

    @model_validator(mode="after")
    def _value_matches_kind(self) -> Self:
        if self.value is None:
            return self
        if self.kind is MetricKind.BOOLEAN and not isinstance(self.value, bool):
            raise TaskContractError(f"boolean metric {self.id} requires true, false, or null")
        if self.kind is MetricKind.CONTINUOUS and (
            isinstance(self.value, bool) or not isinstance(self.value, int | float)
        ):
            raise TaskContractError(f"continuous metric {self.id} requires a number or null")
        if self.kind is MetricKind.CATEGORICAL and not isinstance(self.value, str):
            raise TaskContractError(f"categorical metric {self.id} requires text or null")
        return self


class TrialVector(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    task_version: str
    agent_identity: str
    seed: Annotated[int, Field(ge=0)]
    gates: tuple[GateOutcome, ...]
    metrics: tuple[MetricOutcome, ...]

    @model_validator(mode="after")
    def _one_headline(self) -> Self:
        headlines = [metric for metric in self.metrics if metric.headline]
        if len(headlines) != 1:
            raise TaskContractError(
                f"a trial vector must mark exactly one headline metric, got {len(headlines)}"
            )
        return self

    @property
    def headline(self) -> MetricOutcome:
        return next(metric for metric in self.metrics if metric.headline)

    def metric(self, metric_id: str) -> MetricOutcome | None:
        return next((metric for metric in self.metrics if metric.id == metric_id), None)

    def gate(self, gate_id: str) -> GateOutcome | None:
        return next((gate for gate in self.gates if gate.id == gate_id), None)

    @property
    def any_gate_failed(self) -> bool:
        return any(gate.status is GateStatus.FAIL for gate in self.gates)

    @property
    def any_gate_unassessable(self) -> bool:
        return any(gate.status is GateStatus.NOT_ASSESSABLE for gate in self.gates)

    def __float__(self) -> float:
        raise ScoreContractError(
            "a trial vector has no scalar value; use a pinned declarative projection"
        )

    def __int__(self) -> int:
        return int(self.__float__())

    def __bool__(self) -> bool:
        raise ScoreContractError(
            "a trial vector has no truth value; read a named metric or gate outcome"
        )


def project(vector: TrialVector, spec: ProjectionSpec) -> float:
    """Apply a digestable declarative rule to an authoritative vector."""
    if vector.any_gate_unassessable:
        if spec.gate_unassessable is GateProjectionPolicy.ZERO:
            return spec.false_value
        raise ScoreContractError("cannot project a trial whose gates are unassessable")
    if vector.any_gate_failed:
        if spec.gate_failure is GateProjectionPolicy.ZERO:
            return spec.false_value
        raise ScoreContractError("projection refuses a failed hard gate")
    for metric_id in spec.require_false_metrics:
        outcome = vector.metric(metric_id)
        if outcome is None:
            raise TaskContractError(f"{spec.identity} requires a {metric_id!r} metric")
        if outcome.value is None:
            raise ScoreContractError(f"projection metric {metric_id!r} is unassessable")
        if not isinstance(outcome.value, bool):
            raise TaskContractError(f"projection guard {metric_id!r} must be boolean")
        if outcome.value:
            return spec.false_value
    source = vector.metric(spec.source_metric)
    if source is None:
        raise TaskContractError(f"{spec.identity} requires a {spec.source_metric!r} source metric")
    if source.value is None:
        raise ScoreContractError(f"projection source {spec.source_metric!r} is unassessable")
    if isinstance(source.value, bool):
        return spec.true_value if source.value else spec.false_value
    if isinstance(source.value, int | float):
        return float(source.value)
    raise TaskContractError("categorical metrics cannot be projected to a reward")
