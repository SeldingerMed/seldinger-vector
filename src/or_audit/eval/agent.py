"""An agent is an ``org/name`` package that implements one port.

Harbor: Claude Code, OpenHands, a custom ``BaseAgent``. Here: a policy
checkpoint, a frozen video model, a VLM. Identities are HuggingFace-shaped, so
``seldingermed/lumen-linear`` and ``acme/cabg-vlm`` are both agent packages.
The kernel does not know what CABG is.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from or_audit.errors import TaskContractError
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
    """Loadable agent with an executable, content-pinned package."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: Annotated[str, StringConstraints(min_length=1, max_length=16)]
    id: AgentId
    agent_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    port: PortId
    kind: AgentKind
    weights_pin: str = ""
    weights_path: str = ""
    entrypoint: str = ""

    @model_validator(mode="after")
    def _runtime_contract(self) -> Self:
        if self.kind is AgentKind.RANDOM:
            if self.weights_pin or self.weights_path or self.entrypoint:
                msg = (
                    f"agent {self.id} is kind=random; random baselines cannot "
                    f"pretend to have weights or executable package code"
                )
                raise TaskContractError(msg)
            return self
        if not self.entrypoint:
            raise TaskContractError(f"agent {self.id} must name an entrypoint")
        if not self.weights_path or not self.weights_pin:
            msg = f"agent {self.id} must name content-pinned weights"
            raise TaskContractError(msg)
        return self

    def describe(self) -> str:
        """One block for ``or-audit agents validate``."""
        pin = self.weights_pin if self.weights_pin else "(no weights)"
        return (
            f"Agent {self.id}@{self.agent_version}\n"
            f"  port       {self.port.value}\n"
            f"  kind       {self.kind.value}\n"
            f"  weights    {pin}\n"
            f"  entrypoint {self.entrypoint or '(none)'}"
        )
