"""A dataset is a versioned collection of tasks.

Harbor: a dataset is a collection of tasks, sometimes with custom metrics.
Here the custom metric cannot be 'mean reward', and the headline cannot be
raw reach when safe success exists on any task.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from or_audit.errors import TaskContractError
from or_audit.eval.enums import PhiClass
from or_audit.eval.task import TaskSpec

Slug = Annotated[
    str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
]


class DatasetSpec(BaseModel):
    """Loaded dataset. ``tasks`` are already validated TaskSpecs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: Annotated[str, StringConstraints(min_length=1, max_length=16)]
    id: Slug
    dataset_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    headline: Slug
    phi_class: PhiClass
    tasks: tuple[TaskSpec, ...]
    description: str = ""

    @model_validator(mode="after")
    def _non_empty(self) -> Self:
        if not self.tasks:
            msg = f"dataset {self.id} contains no tasks"
            raise TaskContractError(msg)
        ids = [t.id for t in self.tasks]
        if len(set(ids)) != len(ids):
            msg = f"dataset {self.id} lists the same task id twice"
            raise TaskContractError(msg)
        return self

    def check_tasks(self) -> None:
        """Cross-check tasks against the dataset headline and PHI class.

        Called by the loader after construction so TaskSpecs are in hand.
        """
        for task in self.tasks:
            if task.verifier.headline != self.headline:
                msg = (
                    f"dataset {self.id} headlines {self.headline!r} but task "
                    f"{task.id} headlines {task.verifier.headline!r}"
                )
                raise TaskContractError(msg)
            if task.phi.class_ is not self.phi_class:
                msg = (
                    f"dataset {self.id} is phi={self.phi_class.value} but task "
                    f"{task.id} is phi={task.phi.class_.value}"
                )
                raise TaskContractError(msg)
            metric_ids = {m.id for m in task.verifier.metrics}
            if self.headline == "raw_success" and "safe_success" in metric_ids:
                msg = (
                    f"dataset {self.id} headlines raw_success while task "
                    f"{task.id} measures safe_success; BUILD.md forbids that collapse"
                )
                raise TaskContractError(msg)
