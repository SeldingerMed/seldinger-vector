"""Versioned v0.3 contracts shared by tasks, agents, runners, and scorecards."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from or_audit.errors import TaskContractError

Slug = Annotated[
    str, StringConstraints(min_length=1, max_length=120, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
]


class InteractionMode(StrEnum):
    """Shape of interaction between a harness and an agent."""

    CLOSED_LOOP = "closed-loop"
    SINGLE_TURN = "single-turn"
    INTERACTIVE = "interactive"
    COUNTERFACTUAL = "counterfactual"


class RuntimeKind(StrEnum):
    """Portable agent runtime identities represented by v0.3."""

    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    OPENAI_COMPATIBLE = "openai-compatible"
    CONTAINER = "container"
    TRUSTED_IN_PROCESS = "trusted-in-process"


class MetricKind(StrEnum):
    """Metric value shapes with distinct aggregation rules."""

    BOOLEAN = "boolean"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


class MetricDirection(StrEnum):
    """How a metric should move when results improve."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    NEUTRAL = "neutral"


class GateProjectionPolicy(StrEnum):
    """Projection behavior when a hard gate is not cleanly satisfied."""

    ZERO = "zero"
    REFUSE = "refuse"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class InterfaceSpec(_Frozen):
    """Requirements a task exposes to compatible agents."""

    id: Slug
    interaction_mode: InteractionMode
    protocol_version: str = "1"
    observations: tuple[Slug, ...] = ()
    actions: tuple[Slug, ...] = ()
    outputs: tuple[Slug, ...] = ()
    features: tuple[Slug, ...] = ()
    modalities: tuple[Slug, ...] = ()

    @model_validator(mode="after")
    def _shape_matches_mode(self) -> Self:
        if self.interaction_mode is InteractionMode.CLOSED_LOOP and not self.actions:
            raise TaskContractError(f"closed-loop interface {self.id} must declare an action")
        if self.interaction_mode is not InteractionMode.CLOSED_LOOP and not self.outputs:
            raise TaskContractError(
                f"{self.interaction_mode.value} interface {self.id} needs output"
            )
        return self


class CapabilitySpec(_Frozen):
    """One interface implementation declared by an agent package."""

    interface: Slug
    interaction_modes: tuple[InteractionMode, ...]
    protocol_versions: tuple[str, ...] = ("1",)
    observations: tuple[Slug, ...] = ()
    actions: tuple[Slug, ...] = ()
    outputs: tuple[Slug, ...] = ()
    features: tuple[Slug, ...] = ()
    modalities: tuple[Slug, ...] = ()
    schema_wildcard: bool = False

    @model_validator(mode="after")
    def _non_empty_modes(self) -> Self:
        if not self.interaction_modes:
            raise TaskContractError(f"capability {self.interface} declares no interaction mode")
        return self

    def satisfies(self, interface: InterfaceSpec) -> bool:
        """Return whether this declaration satisfies every task requirement."""
        schemas_match = self.schema_wildcard or (
            set(interface.observations) <= set(self.observations)
            and set(interface.actions) <= set(self.actions)
            and set(interface.outputs) <= set(self.outputs)
            and set(interface.features) <= set(self.features)
            and set(interface.modalities) <= set(self.modalities)
        )
        return (
            self.interface == interface.id
            and interface.interaction_mode in self.interaction_modes
            and interface.protocol_version in self.protocol_versions
            and schemas_match
        )


class HarnessSpec(_Frozen):
    """Task-owned execution mode and protocol limits."""

    interaction_mode: InteractionMode
    protocol_version: str = "1"
    max_steps: Annotated[int, Field(ge=1, le=1_000_000)] = 10_000


class ScenarioSpec(_Frozen):
    """Versioned initial condition for a procedural trial."""

    id: Slug
    version: str = "1"
    description: str = ""
    seed: Annotated[int, Field(ge=0)] = 0
    inputs: dict[str, Any] = Field(default_factory=dict)


class PerturbationSpec(_Frozen):
    """Task-owned disturbance applied to a scenario."""

    id: Slug
    version: str = "1"
    description: str = ""
    scenario_id: Slug | None = None
    kind: Slug
    at_step: Annotated[int, Field(ge=0)] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class RuntimeDescriptor(_Frozen):
    """Pinned description of where and how an agent executes."""

    kind: RuntimeKind
    protocol_version: str = "1"
    command: tuple[str, ...] = ()
    entrypoint: str = ""
    model: str = ""
    revision: str = ""
    base_url: str = ""
    image: str = ""
    image_digest: str = ""
    timeout_sec: Annotated[float, Field(gt=0.0)] = 120.0

    @model_validator(mode="after")
    def _identity_is_pinned(self) -> Self:
        if self.kind in {RuntimeKind.LOCAL, RuntimeKind.TRUSTED_IN_PROCESS}:
            if not self.command and not self.entrypoint:
                raise TaskContractError(f"{self.kind.value} runtime needs command or entrypoint")
        elif self.kind is RuntimeKind.HUGGINGFACE:
            if not self.model or not self.revision:
                raise TaskContractError("huggingface runtime needs model and revision")
        elif self.kind is RuntimeKind.OPENAI_COMPATIBLE:
            if not self.model or not self.base_url:
                raise TaskContractError("openai-compatible runtime needs model and base_url")
        elif self.kind is RuntimeKind.CONTAINER and (not self.image or not self.image_digest):
            raise TaskContractError("container runtime needs image and image_digest")
        return self

    @property
    def identity(self) -> str:
        """Stable runtime identity covered by package and job digests."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def legacy_interface(port: dict[str, Any]) -> InterfaceSpec:
    """Translate a v0.2 port mapping into the canonical v0.3 interface."""
    port_id = str(port.get("id", ""))
    if port_id == "gym-policy":
        return InterfaceSpec(
            id="gym-policy",
            interaction_mode=InteractionMode.CLOSED_LOOP,
            observations=(str(port.get("observation") or "state"),),
            actions=(str(port.get("action") or "continuous-action"),),
        )
    if port_id == "video-predict":
        return InterfaceSpec(
            id="video-predict",
            interaction_mode=InteractionMode.SINGLE_TURN,
            observations=(str(port.get("observation") or "video-clip"),),
            outputs=(str(port.get("prediction") or "structured-prediction"),),
            features=("abstention",),
        )
    raise TaskContractError(f"unknown legacy port {port_id!r}")


def legacy_capability(port_id: str) -> CapabilitySpec:
    """Translate a v0.2 agent port into a capability declaration."""
    if port_id == "gym-policy":
        return CapabilitySpec(
            interface="gym-policy",
            interaction_modes=(InteractionMode.CLOSED_LOOP,),
            schema_wildcard=True,
        )
    if port_id == "video-predict":
        return CapabilitySpec(
            interface="video-predict",
            interaction_modes=(InteractionMode.SINGLE_TURN,),
            schema_wildcard=True,
        )
    raise TaskContractError(f"unknown legacy port {port_id!r}")
