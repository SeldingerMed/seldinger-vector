"""Bind an ``org/name`` agent to a task, or refuse.

The sandbox services ``seldingermed/lumen-linear`` and ``acme/cabg-vlm`` with
the same verb. Compatibility is port (and accepted agent kind), not
procedure name. We do not heuristically "adapt" a video model onto a gym
task.
"""

from __future__ import annotations

from or_audit.errors import TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.task import TaskSpec


def assert_bind(task: TaskSpec, agent: AgentPackage) -> None:
    """Raise unless this agent can be scored on this task.

    Raises:
        TaskContractError: Port mismatch or agent kind the task does not accept.
    """
    if task.port.id is not agent.port:
        msg = (
            f"agent {agent.id} implements {agent.port.value} but task "
            f"{task.id} requires {task.port.id.value}; a video-predict model "
            f"is not silently a gym policy, and the kernel does not invent a "
            f"procedure-specific adapter to paper over that (BUILD.md §1.1a)"
        )
        raise TaskContractError(msg)
    if agent.kind not in task.agent.kinds:
        accepted = ", ".join(k.value for k in task.agent.kinds)
        msg = f"agent {agent.id} is kind={agent.kind.value} but task {task.id} accepts {accepted}"
        raise TaskContractError(msg)
