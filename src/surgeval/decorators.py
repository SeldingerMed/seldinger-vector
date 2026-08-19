"""SurgEval Decorators for Python Models and Tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from or_audit.eval.agent import AgentPackage
from or_audit.eval.contracts import CapabilitySpec, InteractionMode, RuntimeDescriptor, RuntimeKind
from or_audit.eval.enums import AgentKind

T = TypeVar("T")


def agent(
    interface: str = "gym-policy",
    *,
    interaction_mode: InteractionMode | str = InteractionMode.CLOSED_LOOP,
    agent_id: str = "custom-agent",
    version: str = "0",
) -> Callable[[type[T]], type[T]]:
    """Decorator to mark a Python class as a SurgEval-compatible agent."""
    mode = (
        interaction_mode
        if isinstance(interaction_mode, InteractionMode)
        else InteractionMode(interaction_mode)
    )

    def decorator(cls: type[T]) -> type[T]:
        def to_agent_package(override_interface: str | None = None) -> AgentPackage:
            iface = override_interface or interface
            cap = CapabilitySpec(
                interface=iface,
                interaction_modes=(mode,),
                schema_wildcard=True,
            )
            clean_id = agent_id if "/" in agent_id else f"custom/{agent_id}"
            return AgentPackage(
                format_version="1",
                id=clean_id,
                agent_version=version,
                kind=AgentKind.POLICY.value
                if mode is InteractionMode.CLOSED_LOOP
                else AgentKind.FROZEN_MODEL.value,
                capabilities=(cap,),
                weights_path="weights.json",
                weights_pin="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                runtime=RuntimeDescriptor(
                    kind=RuntimeKind.TRUSTED_IN_PROCESS,
                    entrypoint="in_process",
                ),
            )

        cls.to_agent_package = staticmethod(to_agent_package)  # type: ignore[attr-defined]
        return cls

    return decorator
