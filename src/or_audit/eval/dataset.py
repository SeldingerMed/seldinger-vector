"""Versioned tasksets with v0.2 dataset compatibility."""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from or_audit.errors import TaskContractError
from or_audit.eval.enums import PhiClass
from or_audit.eval.task import TaskSpec

Slug = Annotated[
    str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
]
TasksetId = Annotated[
    str,
    StringConstraints(
        min_length=3,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*$",
    ),
]


class TasksetSpec(BaseModel):
    """Canonical collection of validated tasks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: Annotated[str, StringConstraints(min_length=1, max_length=16)]
    id: TasksetId
    taskset_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    headline: Slug
    phi_class: PhiClass
    tasks: tuple[TaskSpec, ...]
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_dataset_name(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        if "taskset_version" not in data and "dataset_version" in data:
            data["taskset_version"] = data.pop("dataset_version")
        return data

    @model_validator(mode="after")
    def _non_empty(self) -> Self:
        if not self.tasks:
            raise TaskContractError(f"taskset {self.id} contains no tasks")
        ids = [task.id for task in self.tasks]
        if len(set(ids)) != len(ids):
            raise TaskContractError(f"taskset {self.id} lists the same task id twice")
        return self

    @property
    def dataset_version(self) -> str:
        """v0.2 compatibility spelling."""
        return self.taskset_version

    def check_tasks(self) -> None:
        for task in self.tasks:
            if task.verifier.headline != self.headline:
                raise TaskContractError(
                    f"taskset {self.id} headlines {self.headline!r} but task "
                    f"{task.id} headlines {task.verifier.headline!r}"
                )
            if task.phi.class_ is not self.phi_class:
                raise TaskContractError(
                    f"taskset {self.id} is phi={self.phi_class.value} but task "
                    f"{task.id} is phi={task.phi.class_.value}"
                )
            metric_ids = {metric.id for metric in task.verifier.metrics}
            if self.headline == "raw_success" and "safe_success" in metric_ids:
                raise TaskContractError(
                    f"taskset {self.id} cannot headline raw_success while "
                    f"task {task.id} measures safe_success"
                )


DatasetSpec = TasksetSpec
DatasetId = TasksetId
