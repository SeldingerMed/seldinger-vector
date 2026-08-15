"""Versioned agent packages with capabilities and portable runtime identity."""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from or_audit.errors import TaskContractError
from or_audit.eval.contracts import (
    CapabilitySpec,
    RuntimeDescriptor,
    RuntimeKind,
    Slug,
    legacy_capability,
)
from or_audit.eval.enums import AgentKind, PortId

AgentId = Annotated[
    str,
    StringConstraints(
        min_length=3,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*$",
    ),
]


class AgentPackage(BaseModel):
    """Canonical v0.3 agent; v0.2 entrypoints normalize during loading."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: Annotated[str, StringConstraints(min_length=1, max_length=16)]
    id: AgentId
    agent_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    kind: Slug
    capabilities: tuple[CapabilitySpec, ...]
    runtime: RuntimeDescriptor | None = None
    port: PortId | None = None
    weights_pin: str = ""
    weights_path: str = ""
    entrypoint: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_v02(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        port = data.get("port")
        if "capabilities" not in data:
            if port is None:
                if str(data.get("kind")) == AgentKind.RANDOM.value:
                    port = PortId.GYM_POLICY.value
                    data["port"] = port
                else:
                    raise TaskContractError("agent requires capabilities or legacy port")
            data["capabilities"] = [legacy_capability(str(port)).model_dump(mode="json")]
        if (
            "runtime" not in data
            and str(data.get("kind")) != AgentKind.RANDOM.value
            and data.get("entrypoint")
        ):
            data["runtime"] = {
                "kind": RuntimeKind.LOCAL.value,
                "entrypoint": str(data["entrypoint"]),
            }
        return data

    @model_validator(mode="after")
    def _runtime_contract(self) -> Self:
        if not self.capabilities:
            raise TaskContractError(f"agent {self.id} must declare at least one capability")
        if self.kind == AgentKind.RANDOM.value:
            if self.weights_pin or self.weights_path or self.entrypoint or self.runtime:
                raise TaskContractError("random baselines cannot declare executable weights")
            return self
        if self.runtime is None:
            raise TaskContractError(f"agent {self.id} must declare a runtime")
        if self.runtime.kind in {
            RuntimeKind.LOCAL,
            RuntimeKind.TRUSTED_IN_PROCESS,
        } and (not self.weights_path or not self.weights_pin):
            raise TaskContractError(f"agent {self.id} must name content-pinned weights")
        return self

    def capability_for(self, interface_id: str) -> CapabilitySpec | None:
        return next((item for item in self.capabilities if item.interface == interface_id), None)

    @property
    def runtime_identity(self) -> str:
        return self.runtime.identity if self.runtime is not None else "builtin-random"

    def describe(self) -> str:
        pin = self.weights_pin if self.weights_pin else "(no weights)"
        capabilities = ", ".join(item.interface for item in self.capabilities)
        runtime = self.runtime.kind.value if self.runtime else "builtin"
        return (
            f"Agent {self.id}@{self.agent_version}\n"
            f"  capabilities {capabilities}\n"
            f"  kind         {self.kind}\n"
            f"  runtime      {runtime}\n"
            f"  weights      {pin}\n"
            f"  entrypoint   {self.entrypoint or '(runtime command)'}"
        )
