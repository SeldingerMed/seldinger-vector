"""Universal Modality Adapter Protocol and Registry.

Every procedural modality (laparoscopy video, CT bronchoscopy, fluoroscopy,
orthopedic kinematics) defines an adapter that normalizes observations,
validates actions, and extracts domain-specific safety telemetry.
"""

from typing import Any, Protocol, runtime_checkable

from or_audit.errors import TaskContractError
from or_audit.eval.enums import ModalityKind


@runtime_checkable
class BaseModalityAdapter(Protocol):
    """Protocol for modality-specific observation and action normalization."""

    modality: ModalityKind | str

    def validate_observation(self, observation: Any) -> bool:
        """Validate that an observation conforms to the modality contract."""
        ...

    def validate_action(self, action: Any) -> bool:
        """Validate that an action conforms to the modality contract."""
        ...

    def preprocess_observation(self, observation: Any) -> Any:
        """Transform raw world observation into agent-facing representation."""
        ...

    def postprocess_action(self, action: Any) -> Any:
        """Transform agent action into simulator/actuator command."""
        ...

    def extract_safety_state(self, step_context: dict[str, Any] | None) -> dict[str, Any]:
        """Extract modality-specific safety indicators from transition context."""
        ...

    def get_schema_spec(self) -> dict[str, Any]:
        """Return JSON-serializable schema description for this modality."""
        ...


class ModalityAdapter:
    modality: ModalityKind | str = ModalityKind.SYNTHETIC_PROCEDURAL

    def validate_observation(self, observation: Any) -> bool:
        """Default observation validator (accepts non-None observations)."""
        return observation is not None

    def validate_action(self, action: Any) -> bool:
        """Default action validator (accepts non-None actions)."""
        return action is not None

    def preprocess_observation(self, observation: Any) -> Any:
        """Default observation preprocessor (pass-through)."""
        return observation

    def postprocess_action(self, action: Any) -> Any:
        """Default action postprocessor (pass-through)."""
        return action

    def extract_safety_state(self, step_context: dict[str, Any] | None) -> dict[str, Any]:
        """Default safety state extractor (returns existing safety dictionary or empty)."""
        if not isinstance(step_context, dict):
            return {}
        info = step_context.get("info")
        safety = step_context.get("safety")
        extracted = dict(safety) if isinstance(safety, dict) else {}
        if isinstance(info, dict):
            for key, value in info.items():
                if key in {"unsafe", "max_pen", "safe_success", "diverged", "force", "radiation"}:
                    extracted.setdefault(key, value)
        return extracted

    def get_schema_spec(self) -> dict[str, Any]:
        """Return schema metadata."""
        modality_str = (
            self.modality.value if isinstance(self.modality, ModalityKind) else str(self.modality)
        )
        return {
            "modality": modality_str,
            "adapter_class": self.__class__.__name__,
        }


_ADAPTER_REGISTRY: dict[str, type[BaseModalityAdapter] | BaseModalityAdapter] = {}


def register_adapter(
    modality: ModalityKind | str,
    adapter: type[BaseModalityAdapter] | BaseModalityAdapter,
    *,
    override: bool = False,
) -> None:
    """Register an adapter class or instance for a given modality."""
    key = modality.value if isinstance(modality, ModalityKind) else str(modality)
    if key in _ADAPTER_REGISTRY and not override:
        raise TaskContractError(f"modality adapter already registered for {key!r}")
    _ADAPTER_REGISTRY[key] = adapter


def get_adapter(modality: ModalityKind | str, **kwargs: Any) -> BaseModalityAdapter | None:
    """Get an instantiated adapter for a given modality, or None if not registered."""
    key = modality.value if isinstance(modality, ModalityKind) else str(modality)
    entry = _ADAPTER_REGISTRY.get(key)
    if entry is None:
        return None
    if isinstance(entry, type):
        return entry(**kwargs)
    return entry


def require_adapter(modality: ModalityKind | str, **kwargs: Any) -> BaseModalityAdapter:
    """Get an adapter or raise TaskContractError if not registered."""
    adapter = get_adapter(modality, **kwargs)
    if adapter is None:
        key = modality.value if isinstance(modality, ModalityKind) else str(modality)
        known = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
        raise TaskContractError(f"unknown modality {key!r}; registered adapters: {known}")
    return adapter


def list_adapters() -> dict[str, str]:
    """Return dictionary of registered modalities and their adapter class names."""
    return {
        k: v.__name__ if isinstance(v, type) else v.__class__.__name__
        for k, v in sorted(_ADAPTER_REGISTRY.items())
    }


def clear_registry() -> None:
    """Reset the adapter registry (primarily for test isolation)."""
    _ADAPTER_REGISTRY.clear()
