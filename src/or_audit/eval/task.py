"""The Harbor task, with the fields medicine actually needs.

A Harbor task is instruction + container + tests that write ``reward.txt``.
A task here is instruction + world + vector verifier. The directory layout
is deliberately Harbor-shaped so the analog is obvious:

```
<task>/
  task.toml
  instruction.md
  verifier.toml     # optional; the RL projection lives here, not in the vector
```
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from or_audit.errors import TaskContractError
from or_audit.eval.enums import (
    AgentKind,
    AttestationLevel,
    OracleKind,
    PhiClass,
    ProjectionId,
    SubjectKind,
    WorldKind,
)

Slug = Annotated[
    str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
]
NonEmpty = Annotated[str, StringConstraints(min_length=1, max_length=200)]
Instruction = Annotated[str, StringConstraints(min_length=1, max_length=20_000)]


class _Frozen(BaseModel):
    """Immutable task records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class TaskMetadata(_Frozen):
    """Human-facing description. Not load-bearing."""

    title: NonEmpty
    modality: NonEmpty
    tags: tuple[str, ...] = ()
    safety_critical: bool = True


class SubjectSpec(_Frozen):
    """Who the trial scores."""

    kind: SubjectKind


class PhiSpec(_Frozen):
    """Isolation class of the world."""

    class_: PhiClass = Field(alias="class")

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class WorldSpec(_Frozen):
    """The procedural world. Harbor's Dockerfile analogue."""

    kind: WorldKind
    gym_id: str = ""
    world_pin: str = ""
    n_eval_episodes: Annotated[int, Field(ge=1, le=10_000)] = 30
    seed_policy: str = "deterministic-eval-30"

    @model_validator(mode="after")
    def _lumen_has_gym_id(self) -> Self:
        if self.kind is WorldKind.LUMEN_GYM and not self.gym_id:
            msg = "a lumen-gym world must name gym_id (e.g. Lumen/NavTreeBranch-v0)"
            raise TaskContractError(msg)
        return self


class AgentSpec(_Frozen):
    """Which agent kinds this task accepts."""

    kinds: tuple[AgentKind, ...]
    action_space: str = ""
    timeout_sec: Annotated[float, Field(gt=0.0)] = 120.0

    @model_validator(mode="after")
    def _at_least_one_kind(self) -> Self:
        if not self.kinds:
            msg = "a task must accept at least one agent kind"
            raise TaskContractError(msg)
        return self


class OracleSpec(_Frozen):
    """Where ground truth comes from."""

    kind: OracleKind


class GateSpec(_Frozen):
    """One hard gate the verifier will report."""

    id: Slug
    source: str = ""
    fail_when: str = ""
    maps_to: str = ""


class MetricSpec(_Frozen):
    """One metric the verifier will report. Not a gate."""

    id: Slug
    source: str = ""


class VerifierSpec(_Frozen):
    """Vector verifier. Headline is required and must be a metric id."""

    abstain_ok: bool
    headline: Slug
    gates: tuple[GateSpec, ...] = ()
    metrics: tuple[MetricSpec, ...] = ()

    @model_validator(mode="after")
    def _headline_is_a_metric(self) -> Self:
        metric_ids = [m.id for m in self.metrics]
        if len(set(metric_ids)) != len(metric_ids):
            msg = "verifier metrics must have unique ids"
            raise TaskContractError(msg)
        gate_ids = [g.id for g in self.gates]
        if len(set(gate_ids)) != len(gate_ids):
            msg = "verifier gates must have unique ids"
            raise TaskContractError(msg)
        if self.headline not in metric_ids:
            msg = (
                f"headline {self.headline!r} is not a declared metric; the "
                f"headline must be one of the vector's metrics, not a gate "
                f"and not an implicit scalar"
            )
            raise TaskContractError(msg)
        return self


class AttestationSpec(_Frozen):
    """Whether this task mints de-id attestations."""

    level: AttestationLevel = AttestationLevel.NONE


class DecisionSpec(_Frozen):
    """Human-subject determinations. Refused on the eval wedge."""

    emit_human_determination: bool = False


class ProjectionSpec(_Frozen):
    """Optional RL collapse. Lives beside the vector, never instead of it."""

    id: ProjectionId
    version: Annotated[str, StringConstraints(min_length=1, max_length=32)] = "0"


class TaskSpec(_Frozen):
    """A loadable eval task."""

    format_version: Annotated[str, StringConstraints(min_length=1, max_length=16)]
    id: Slug
    task_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    metadata: TaskMetadata
    subject: SubjectSpec
    phi: PhiSpec
    environment: WorldSpec
    agent: AgentSpec
    oracle: OracleSpec
    verifier: VerifierSpec
    attestation: AttestationSpec = AttestationSpec()
    decision: DecisionSpec = DecisionSpec()
    instruction: Instruction
    projection: ProjectionSpec | None = None

    @model_validator(mode="after")
    def _invariants(self) -> Self:
        if self.phi.class_ is PhiClass.PROHIBITED:
            msg = f"task {self.id} is marked phi=prohibited and cannot be loaded"
            raise TaskContractError(msg)
        if self.phi.class_ is PhiClass.PROCEDURAL and (
            self.attestation.level is not AttestationLevel.NONE
        ):
            msg = (
                f"task {self.id} is procedural geometry; dressing it in "
                f"attestation language is how a sim eval pretends to be a "
                f"clinical release"
            )
            raise TaskContractError(msg)
        if self.decision.emit_human_determination:
            msg = (
                f"task {self.id} asks to emit a human determination; BUILD.md "
                f"P0 refuses that path until PLAN.md Phase 0 clears, including "
                f"when subject.kind is human"
            )
            raise TaskContractError(msg)
        if self.subject.kind is SubjectKind.HUMAN:
            msg = (
                f"task {self.id} has subject.kind=human; the eval wedge scores "
                f"policies and models (BUILD.md §1.3)"
            )
            raise TaskContractError(msg)
        if self.metadata.safety_critical and not self.verifier.gates:
            msg = (
                f"task {self.id} is safety_critical but declares no gates; a "
                f"reach-only task is how raw success hides wall injury"
            )
            raise TaskContractError(msg)
        if self.oracle.kind is OracleKind.PHYSICS and self.environment.kind not in {
            WorldKind.LUMEN_GYM,
            WorldKind.LUMEN_REPLAY,
        }:
            msg = (
                f"task {self.id} claims a physics oracle but world kind is "
                f"{self.environment.kind.value}"
            )
            raise TaskContractError(msg)
        metric_ids = {m.id for m in self.verifier.metrics}
        if "safe_success" in metric_ids and self.verifier.headline == "raw_success":
            msg = (
                f"task {self.id} declares safe_success but headlines raw_success; "
                f"that is the CathSim failure mode BUILD.md forbids"
            )
            raise TaskContractError(msg)
        return self

    def assert_runnable(self) -> None:
        """Raise unless the world is pinned enough to replay.

        Validate-only tasks (no pin yet) are loadable. They are not runnable.
        """
        if self.environment.kind is WorldKind.LUMEN_GYM and not self.environment.world_pin:
            msg = (
                f"task {self.id} has no world_pin; P1 evals must pin a Lumen "
                f"commit so a leaderboard row can replay"
            )
            raise TaskContractError(msg)

    def describe(self) -> str:
        """One block for ``or-audit tasks describe``."""
        pin = self.environment.world_pin or "(unpinned — not runnable)"
        projection = (
            f"{self.projection.id.value}@{self.projection.version}" if self.projection else "none"
        )
        gates = ", ".join(g.id for g in self.verifier.gates) or "(none)"
        metrics = ", ".join(m.id for m in self.verifier.metrics)
        return (
            f"Task {self.id}@{self.task_version} ({self.metadata.title})\n"
            f"  world      {self.environment.kind.value} {self.environment.gym_id} pin={pin}\n"
            f"  subject    {self.subject.kind.value}  phi={self.phi.class_.value}\n"
            f"  oracle     {self.oracle.kind.value}\n"
            f"  agents     {', '.join(k.value for k in self.agent.kinds)}\n"
            f"  gates      {gates}\n"
            f"  metrics    {metrics}\n"
            f"  headline   {self.verifier.headline}\n"
            f"  projection {projection}\n"
            f"  human det. refused"
        )
