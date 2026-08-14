"""Load Harbor-shaped task, dataset, and agent directories."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from or_audit.errors import TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.dataset import DatasetSpec
from or_audit.eval.integrity import file_sha256, package_file
from or_audit.eval.task import TaskSpec


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"missing {path.name}: {path}"
        raise TaskContractError(msg)
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _task_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_dir():
        return resolved
    if resolved.name == "task.toml":
        return resolved.parent
    msg = f"a task is a directory (or its task.toml), got {path}"
    raise TaskContractError(msg)


def load_task(path: Path | str) -> TaskSpec:
    """Load and validate a task directory.

    Raises:
        TaskContractError: If files are missing or the contract fails.
    """
    root = _task_root(Path(path))
    data = _read_toml(root / "task.toml")
    instruction_path = root / "instruction.md"
    if not instruction_path.is_file():
        msg = f"task {root} is missing instruction.md"
        raise TaskContractError(msg)
    data["instruction"] = instruction_path.read_text(encoding="utf-8").strip()
    verifier_path = root / "verifier.toml"
    if verifier_path.is_file():
        extra = _read_toml(verifier_path)
        if "projection" in extra:
            data["projection"] = extra["projection"]
    try:
        return TaskSpec.model_validate(data)
    except TaskContractError:
        raise
    except Exception as exc:
        msg = f"task {root} failed validation: {exc}"
        raise TaskContractError(msg) from exc


def load_dataset(path: Path | str) -> DatasetSpec:
    """Load a dataset and every task it names.

    Task paths in ``dataset.toml`` are resolved relative to the dataset
    directory.
    """
    dataset_path = Path(path).resolve()
    root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    toml_path = root / "dataset.toml" if dataset_path.is_dir() else dataset_path
    data = _read_toml(toml_path)
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        msg = f"dataset {toml_path} must list at least one task path"
        raise TaskContractError(msg)
    tasks = tuple(load_task((root / str(entry)).resolve()) for entry in raw_tasks)
    payload = {k: v for k, v in data.items() if k != "tasks"}
    try:
        spec = DatasetSpec.model_validate({**payload, "tasks": tasks})
    except TaskContractError:
        raise
    except Exception as exc:
        msg = f"dataset {toml_path} failed validation: {exc}"
        raise TaskContractError(msg) from exc
    spec.check_tasks()
    return spec


def dataset_task_paths(path: Path | str) -> tuple[Path, ...]:
    """Task directories listed by a dataset, resolved against the dataset root."""
    dataset_path = Path(path).resolve()
    root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    toml_path = root / "dataset.toml" if dataset_path.is_dir() else dataset_path
    data = _read_toml(toml_path)
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        msg = f"dataset {toml_path} must list at least one task path"
        raise TaskContractError(msg)
    return tuple((root / str(entry)).resolve() for entry in raw_tasks)


def _agent_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_dir():
        return resolved
    if resolved.name == "agent.toml":
        return resolved.parent
    msg = f"an agent is a directory (or its agent.toml), got {path}"
    raise TaskContractError(msg)


def load_agent(path: Path | str) -> AgentPackage:
    """Load and validate an ``org/name`` agent package.

    Raises:
        TaskContractError: If files are missing or the contract fails.
    """
    root = _agent_root(Path(path))
    data = _read_toml(root / "agent.toml")
    try:
        agent = AgentPackage.model_validate(data)
        if agent.weights_path:
            weights = package_file(root, agent.weights_path, label="agent weights")
            actual = file_sha256(weights)
            if actual != agent.weights_pin:
                msg = (
                    f"agent {agent.id} weights digest mismatch: "
                    f"declared {agent.weights_pin}, actual {actual}"
                )
                raise TaskContractError(msg)
        return agent
    except TaskContractError:
        raise
    except Exception as exc:
        msg = f"agent {root} failed validation: {exc}"
        raise TaskContractError(msg) from exc
