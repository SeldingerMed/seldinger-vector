"""Task-owned v0.3 evaluation contracts with deterministic v0.2 normalization."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from or_audit.errors import TaskContractError
from or_audit.eval.contracts import (
    GateProjectionPolicy,
    HarnessSpec,
    InteractionMode,
    InterfaceSpec,
    MetricDirection,
    MetricKind,
    PerturbationSpec,
    ScenarioSpec,
    legacy_interface,
)
from or_audit.eval.enums import (
    AttestationLevel,
    OracleKind,
    PhiClass,
    PortId,
    ProjectionId,
    SubjectKind,
    WorldKind,
)

Slug = Annotated[
    str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
]
NonEmpty = Annotated[str, StringConstraints(min_length=1, max_length=200)]
Instruction = Annotated[str, StringConstraints(min_length=1, max_length=20_000)]

_BOOLEAN_METRICS = {
    "abstained",
    "diverged",
    "next_step_correct",
    "outcome_correct",
    "raw_success",
    "release_audit_passed",
    "safe_success",
    "failure_detected",
    "recovered",
    "safe_abandonment",
    "unsafe_persistence",
    "harm_after_failure",
    "handoff_accepted",
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TaskMetadata(_Frozen):
    """Human-facing metadata; tags remain open search terms."""

    title: NonEmpty
    modality: NonEmpty
    tags: tuple[str, ...] = ()
    safety_critical: bool = True


class PortSpec(_Frozen):
    """Deprecated v0.2 port retained only as a compatibility input."""

    id: PortId
    observation: str = ""
    action: str = ""
    prediction: str = ""

    @model_validator(mode="after")
    def _video_predict_names_a_schema(self) -> Self:
        if self.id is PortId.VIDEO_PREDICT and not self.prediction:
            raise TaskContractError("a video-predict port must name prediction")
        return self


class SubjectSpec(_Frozen):
    kind: SubjectKind


class PhiSpec(_Frozen):
    class_: PhiClass = Field(alias="class")
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class WorldSpec(_Frozen):
    """Pinned procedural world and its task-owned inputs."""

    kind: WorldKind
    gym_id: str = ""
    world_pin: str = ""
    parameters: dict[str, bool | int | float | str] = Field(default_factory=dict)
    n_eval_episodes: Annotated[int, Field(ge=1, le=10_000)] = 30
    seed_policy: str = "deterministic-eval-30"
    inputs_path: str = ""
    labels_path: str = ""
    contract_path: str = ""

    @model_validator(mode="after")
    def _required_paths(self) -> Self:
        if self.kind in {WorldKind.LUMEN_GYM, WorldKind.GYM} and not self.gym_id:
            raise TaskContractError(f"a {self.kind.value} world must name gym_id")
        if self.kind is WorldKind.ANGIOSTRESS_CONTRACT and not self.contract_path:
            raise TaskContractError("an angiostress-contract world must name contract_path")
        return self


class AgentSpec(_Frozen):
    kinds: tuple[Slug, ...]
    action_space: str = ""
    timeout_sec: Annotated[float, Field(gt=0.0)] = 120.0

    @model_validator(mode="after")
    def _at_least_one_kind(self) -> Self:
        if not self.kinds:
            raise TaskContractError("a task must accept at least one agent kind")
        return self


class OracleSpec(_Frozen):
    kind: OracleKind


class GateSpec(_Frozen):
    id: Slug
    source: str = ""
    fail_when: str = ""
    maps_to: str = ""


class MetricSpec(_Frozen):
    """Typed metric declaration with kind-specific aggregation metadata."""

    id: Slug
    source: str = ""
    kind: MetricKind = MetricKind.CONTINUOUS
    unit: str = ""
    direction: MetricDirection = MetricDirection.NEUTRAL
    categories: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _categorical_has_categories(self) -> Self:
        if self.kind is MetricKind.CATEGORICAL and not self.categories:
            raise TaskContractError(f"categorical metric {self.id} must declare categories")
        return self


class VerifierSpec(_Frozen):
    abstain_ok: bool
    headline: Slug
    gates: tuple[GateSpec, ...] = ()
    metrics: tuple[MetricSpec, ...] = ()
    entrypoint: str = ""

    @model_validator(mode="after")
    def _shape(self) -> Self:
        metric_ids = [metric.id for metric in self.metrics]
        gate_ids = [gate.id for gate in self.gates]
        if len(set(metric_ids)) != len(metric_ids):
            raise TaskContractError("verifier metrics must have unique ids")
        if len(set(gate_ids)) != len(gate_ids):
            raise TaskContractError("verifier gates must have unique ids")
        if self.headline not in metric_ids:
            raise TaskContractError(
                f"headline {self.headline!r} is not a declared metric; "
                "the headline cannot be an implicit scalar"
            )
        return self


class AttestationSpec(_Frozen):
    level: AttestationLevel = AttestationLevel.NONE


class DecisionSpec(_Frozen):
    emit_human_determination: bool = False


class ProjectionSpec(_Frozen):
    """Versioned declarative projection recomputed from an authoritative vector."""

    id: ProjectionId | Slug
    version: Annotated[str, StringConstraints(min_length=1, max_length=32)] = "0"
    source_metric: Slug = "raw_success"
    require_false_metrics: tuple[Slug, ...] = ("diverged",)
    gate_failure: GateProjectionPolicy = GateProjectionPolicy.ZERO
    gate_unassessable: GateProjectionPolicy = GateProjectionPolicy.REFUSE
    true_value: float = 1.0
    false_value: float = 0.0

    @field_validator("id", mode="before")
    @classmethod
    def _known_projection_id(cls, value: Any) -> Any:
        try:
            return ProjectionId(value)
        except ValueError:
            return value

    @property
    def rule_digest(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def identity(self) -> str:
        projection_id = self.id.value if isinstance(self.id, ProjectionId) else self.id
        return f"{projection_id}@{self.version}+{self.rule_digest}"


class TaskSpec(_Frozen):
    """Canonical task model; v0.2 port packages normalize before validation."""

    format_version: Annotated[str, StringConstraints(min_length=1, max_length=16)]
    id: Slug
    task_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    metadata: TaskMetadata
    subject: SubjectSpec
    phi: PhiSpec
    environment: WorldSpec
    interface: InterfaceSpec
    harness: HarnessSpec
    scenarios: tuple[ScenarioSpec, ...] = ()
    perturbations: tuple[PerturbationSpec, ...] = ()
    port: PortSpec | None = None
    agent: AgentSpec
    oracle: OracleSpec
    verifier: VerifierSpec
    attestation: AttestationSpec = AttestationSpec()
    decision: DecisionSpec = DecisionSpec()
    instruction: Instruction
    projection: ProjectionSpec | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_v02(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        port = data.get("port")
        if "interface" not in data:
            if not isinstance(port, dict):
                raise TaskContractError("task requires interface or legacy port")
            data["interface"] = legacy_interface(port).model_dump(mode="json")
        if "harness" not in data:
            interface = InterfaceSpec.model_validate(data["interface"])
            data["harness"] = {"interaction_mode": interface.interaction_mode.value}
        verifier = data.get("verifier")
        if isinstance(verifier, dict):
            normalized_verifier = dict(verifier)
            metrics = []
            for raw_metric in normalized_verifier.get("metrics", []):
                metric = dict(raw_metric)
                if "kind" not in metric:
                    metric["kind"] = (
                        MetricKind.BOOLEAN.value
                        if str(metric.get("id")) in _BOOLEAN_METRICS
                        else MetricKind.CONTINUOUS.value
                    )
                metrics.append(metric)
            normalized_verifier["metrics"] = metrics
            data["verifier"] = normalized_verifier
        return data

    @model_validator(mode="after")
    def _invariants(self) -> Self:
        if self.harness.interaction_mode is not self.interface.interaction_mode:
            raise TaskContractError("harness interaction mode must match the task interface")
        scenario_ids = [scenario.id for scenario in self.scenarios]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise TaskContractError(f"task {self.id} has duplicate scenario ids")
        perturbation_ids = [perturbation.id for perturbation in self.perturbations]
        if len(set(perturbation_ids)) != len(perturbation_ids):
            raise TaskContractError(f"task {self.id} has duplicate perturbation ids")
        unknown_scenarios = {
            perturbation.scenario_id
            for perturbation in self.perturbations
            if perturbation.scenario_id is not None and perturbation.scenario_id not in scenario_ids
        }
        if unknown_scenarios:
            raise TaskContractError(
                f"task {self.id} perturbations reference unknown scenarios "
                f"{sorted(unknown_scenarios)}"
            )
        if self.interface.interaction_mode is InteractionMode.CLOSED_LOOP:
            scenario_seeds = [scenario.seed for scenario in self.scenarios]
            if len(set(scenario_seeds)) != len(scenario_seeds):
                raise TaskContractError(
                    f"closed-loop task {self.id} maps more than one scenario to a seed"
                )
        if any(
            perturbation.at_step is not None and perturbation.at_step >= self.harness.max_steps
            for perturbation in self.perturbations
        ):
            raise TaskContractError(
                f"task {self.id} schedules a perturbation outside harness max_steps"
            )
        if self.phi.class_ is PhiClass.PROHIBITED:
            raise TaskContractError(f"task {self.id} is marked phi=prohibited and cannot load")
        if self.phi.class_ is PhiClass.PROCEDURAL and (
            self.attestation.level is not AttestationLevel.NONE
        ):
            raise TaskContractError("procedural geometry cannot mint a clinical attestation")
        if self.decision.emit_human_determination or self.subject.kind is SubjectKind.HUMAN:
            raise TaskContractError(
                "subject.kind=human and human determinations are outside this eval framework"
            )
        if self.metadata.safety_critical and not self.verifier.gates:
            raise TaskContractError(f"safety_critical task {self.id} must declare hard gates")
        if self.oracle.kind is OracleKind.PHYSICS and self.environment.kind not in {
            WorldKind.LUMEN_GYM,
            WorldKind.LUMEN_REPLAY,
            WorldKind.GYM,
        }:
            raise TaskContractError("a physics oracle requires a gym or replay world")
        if self.interface.interaction_mode is InteractionMode.CLOSED_LOOP and (
            self.environment.kind
            not in {WorldKind.LUMEN_GYM, WorldKind.LUMEN_REPLAY, WorldKind.GYM}
        ):
            raise TaskContractError(
                f"{self.interface.id} closed-loop tasks require a gym or replay world"
            )
        if self.interface.interaction_mode is InteractionMode.COUNTERFACTUAL and (
            self.environment.kind is not WorldKind.COUNTERFACTUAL
        ):
            raise TaskContractError("counterfactual interfaces require a counterfactual world")
        metric_ids = {metric.id for metric in self.verifier.metrics}
        if "safe_success" in metric_ids and self.verifier.headline == "raw_success":
            raise TaskContractError(
                "CathSim failure mode: safe_success cannot be hidden behind raw_success"
            )
        return self

    def assert_runnable(self) -> None:
        if self.environment.kind in {WorldKind.LUMEN_GYM, WorldKind.GYM} and not (
            self.environment.world_pin
        ):
            raise TaskContractError(f"task {self.id} has no world_pin")
        if not self.verifier.entrypoint:
            raise TaskContractError(f"task {self.id} has no verifier entrypoint")
        if self.interface.interaction_mode is not InteractionMode.CLOSED_LOOP:
            if not self.environment.inputs_path:
                raise TaskContractError(f"task {self.id} has no inputs_path")
            if not self.environment.labels_path:
                raise TaskContractError(f"task {self.id} has no labels_path")

    def metric(self, metric_id: str) -> MetricSpec:
        metric = next((item for item in self.verifier.metrics if item.id == metric_id), None)
        if metric is None:
            raise TaskContractError(f"task {self.id} does not declare metric {metric_id}")
        return metric

    def describe(self) -> str:
        pin = self.environment.world_pin or "(unpinned)"
        projection = self.projection.identity if self.projection else "none"
        gates = ", ".join(gate.id for gate in self.verifier.gates) or "(none)"
        metrics = ", ".join(metric.id for metric in self.verifier.metrics)
        return (
            f"Task {self.id}@{self.task_version} ({self.metadata.title})\n"
            f"  interface  {self.interface.id} ({self.harness.interaction_mode.value})\n"
            f"  port       {self.port.id.value if self.port else self.interface.id}\n"
            f"  world      {self.environment.kind.value} pin={pin}\n"
            f"  subject    {self.subject.kind.value}  phi={self.phi.class_.value}"
            "  human det. refused\n"
            f"  oracle     {self.oracle.kind.value}\n"
            f"  agents     {', '.join(self.agent.kinds)}\n"
            f"  gates      {gates}\n"
            f"  metrics    {metrics}\n"
            f"  headline   {self.verifier.headline}\n"
            f"  projection {projection}"
        )
