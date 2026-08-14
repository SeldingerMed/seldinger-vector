"""Load task and agent entrypoints from versioned package directories."""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast, runtime_checkable

from or_audit.errors import TaskContractError
from or_audit.eval.integrity import package_file


@runtime_checkable
class PolicyRuntime(Protocol):
    """A loaded gym policy."""

    def reset(self, *, seed: int) -> None:
        """Reset episode-local state."""

    def act(self, observation: Any, *, step: int) -> Any:
        """Return one action for the current observation."""


@runtime_checkable
class PredictorRuntime(Protocol):
    """A loaded video/contract predictor."""

    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        """Return one structured prediction without receiving its label."""


@runtime_checkable
class VerifierRuntime(Protocol):
    """A task-owned vector verifier."""

    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        """Return declared gate outcomes and metric values."""


def _module(path: Path) -> ModuleType:
    module_name = f"or_audit_plugin_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise TaskContractError(f"cannot load plugin module {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise TaskContractError(f"plugin {path} failed to import: {exc}") from exc
    return module


def load_entrypoint(root: Path, entrypoint: str, *, label: str) -> Callable[..., Any]:
    """Load ``relative.py:symbol`` from ``root`` without changing ``sys.path``."""
    module_path, separator, symbol = entrypoint.partition(":")
    if not separator or not module_path or not symbol:
        msg = f"{label} entrypoint must be relative.py:symbol, got {entrypoint!r}"
        raise TaskContractError(msg)
    path = package_file(root, module_path, label=f"{label} module")
    target = getattr(_module(path), symbol, None)
    if not callable(target):
        raise TaskContractError(f"{label} entrypoint {entrypoint!r} is not callable")
    return cast(Callable[..., Any], target)


def load_policy_runtime(root: Path, entrypoint: str, weights_path: str) -> PolicyRuntime:
    factory = load_entrypoint(root, entrypoint, label="policy")
    runtime = factory(root=root, weights_path=package_file(root, weights_path, label="weights"))
    if not isinstance(runtime, PolicyRuntime):
        raise TaskContractError("policy factory must return an object with reset(seed=) and act()")
    return runtime


def load_predictor_runtime(root: Path, entrypoint: str, weights_path: str) -> PredictorRuntime:
    factory = load_entrypoint(root, entrypoint, label="predictor")
    runtime = factory(root=root, weights_path=package_file(root, weights_path, label="weights"))
    if not isinstance(runtime, PredictorRuntime):
        raise TaskContractError("predictor factory must return an object with predict(item)")
    return runtime


def load_verifier_runtime(root: Path, entrypoint: str) -> VerifierRuntime:
    factory = load_entrypoint(root, entrypoint, label="verifier")
    runtime = factory(root=root)
    if not isinstance(runtime, VerifierRuntime):
        raise TaskContractError("verifier factory must return an object with score(context)")
    return runtime
