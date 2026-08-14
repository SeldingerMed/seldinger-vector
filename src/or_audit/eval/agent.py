"""An agent is an ``org/name`` package that implements one port.

Harbor: Claude Code, OpenHands, a custom ``BaseAgent``. Here: a policy
checkpoint, a frozen video model, a VLM. The identity is HuggingFace-shaped
so ``seldingermed/cathmodel`` and ``acme/cabg-vlm`` are the same kind of
object. The kernel does not know what CABG is.
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
    """Loadable agent. Weights and entrypoints are pinned later; P0 is the contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: Annotated[str, StringConstraints(min_length=1, max_length=16)]
    id: AgentId
    agent_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    port: PortId
    kind: AgentKind
    weights_pin: str = ""
    entrypoint: str = ""
    #: Relative to the agent directory. Frozen JSON predictions for P2.
    predictions_path: str = ""

    @model_validator(mode="after")
    def _random_needs_no_weights(self) -> Self:
        if self.kind is AgentKind.RANDOM and self.weights_pin:
            msg = (
                f"agent {self.id} is kind=random; a weights pin would pretend "
                f"a baseline is a checkpoint"
            )
            raise TaskContractError(msg)
        return self

    def describe(self) -> str:
        """One block for ``or-audit agents validate``."""
        pin = self.weights_pin or "(no weights pin)"
        return (
            f"Agent {self.id}@{self.agent_version}\n"
            f"  port       {self.port.value}\n"
            f"  kind       {self.kind.value}\n"
            f"  weights    {pin}\n"
            f"  entrypoint {self.entrypoint or '(none)'}"
        )
