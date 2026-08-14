"""Trial result vector. The thing Harbor would put in reward.txt.

A trial produces gates and metrics. It does not produce a number. The only
sanctioned float is a :class:`~or_audit.eval.projection.ProjectionSpec`
applied by :func:`project`, and that float is for RL, not for the leaderboard.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from or_audit.domain.enums import GateStatus
from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.enums import ProjectionId
from or_audit.eval.task import ProjectionSpec


class GateOutcome(BaseModel):
    """One hard gate on one trial."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    status: GateStatus
    reason: str = ""


class MetricOutcome(BaseModel):
    """One metric on one trial. ``None`` means unassessable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    value: bool | float | None
    headline: bool = False


class TrialVector(BaseModel):
    """Gates and metrics for one trial, kept apart."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    task_version: str
    agent_identity: str
    seed: Annotated[int, Field(ge=0)]
    gates: tuple[GateOutcome, ...]
    metrics: tuple[MetricOutcome, ...]

    @model_validator(mode="after")
    def _one_headline(self) -> Self:
        headlines = [m for m in self.metrics if m.headline]
        if len(headlines) != 1:
            msg = (
                f"a trial vector must mark exactly one headline metric, got "
                f"{len(headlines)}; this is how raw reach is stopped from "
                f"standing in for safe success"
            )
            raise TaskContractError(msg)
        return self

    @property
    def headline(self) -> MetricOutcome:
        """The dataset's headline metric for this trial."""
        return next(m for m in self.metrics if m.headline)

    def metric(self, metric_id: str) -> MetricOutcome | None:
        """Named metric, if present."""
        return next((m for m in self.metrics if m.id == metric_id), None)

    def gate(self, gate_id: str) -> GateOutcome | None:
        """Named gate, if present."""
        return next((g for g in self.gates if g.id == gate_id), None)

    @property
    def any_gate_failed(self) -> bool:
        """Whether a hard gate affirmatively failed."""
        return any(g.status is GateStatus.FAIL for g in self.gates)

    @property
    def any_gate_unassessable(self) -> bool:
        """Whether a hard gate could not be decided."""
        return any(g.status is GateStatus.NOT_ASSESSABLE for g in self.gates)

    def __float__(self) -> float:
        """Always raises. Use :func:`project` for an RL float."""
        msg = (
            "a trial vector has no scalar value; the leaderboard reports gates "
            "and metrics, and RL may only see a versioned projection "
            "(BUILD.md §1.3)"
        )
        raise ScoreContractError(msg)

    def __int__(self) -> int:
        """Always raises."""
        return int(self.__float__())

    def __bool__(self) -> bool:
        """Always raises so ``if vector`` cannot mean 'did it succeed'."""
        msg = (
            "a trial vector has no truth value; read headline.value or "
            "any_gate_failed so the call site names the question"
        )
        raise ScoreContractError(msg)


def project(vector: TrialVector, spec: ProjectionSpec) -> float:
    """Collapse ``vector`` to a float for RL.

    Raises:
        ScoreContractError: If a gate is unassessable (missing evidence is
            not a zero reward).
        TaskContractError: If required metrics are absent.
    """
    if vector.any_gate_unassessable:
        msg = (
            "cannot project a trial whose gates are unassessable; missing "
            "evidence is not a reward of zero"
        )
        raise ScoreContractError(msg)
    match spec.id:
        case ProjectionId.GATED_REACH_V0:
            if vector.any_gate_failed:
                return 0.0
            diverged = vector.metric("diverged")
            if diverged is None:
                msg = "gated_reach_v0 requires a 'diverged' metric"
                raise TaskContractError(msg)
            if diverged.value is True:
                return 0.0
            raw = vector.metric("raw_success")
            if raw is None or not isinstance(raw.value, bool):
                msg = "gated_reach_v0 requires a boolean 'raw_success' metric"
                raise TaskContractError(msg)
            return 1.0 if raw.value else 0.0
