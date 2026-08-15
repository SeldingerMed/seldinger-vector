"""Bind task interface requirements to declared agent capabilities."""

from __future__ import annotations

from or_audit.errors import TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.task import TaskSpec


def assert_bind(task: TaskSpec, agent: AgentPackage) -> None:
    """Raise unless one capability satisfies the complete task interface."""
    capability = next(
        (candidate for candidate in agent.capabilities if candidate.satisfies(task.interface)),
        None,
    )
    if capability is None:
        declared = ", ".join(item.interface for item in agent.capabilities)
        raise TaskContractError(
            f"agent {agent.id} capabilities [{declared}] do not satisfy "
            f"task {task.id} interface {task.interface.id} "
            f"({task.interface.interaction_mode.value}); legacy ports such as "
            "video-predict and gym-policy are normalized before this check"
        )
    if agent.kind not in task.agent.kinds:
        accepted = ", ".join(task.agent.kinds)
        raise TaskContractError(
            f"agent {agent.id} is kind={agent.kind} but task {task.id} accepts {accepted}"
        )
