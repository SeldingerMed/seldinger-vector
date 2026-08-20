"""Universal Modality Adapter Protocol and Registry.

Every procedural modality (laparoscopy video, CT bronchoscopy, fluoroscopy,
orthopedic kinematics) defines an adapter that normalizes observations,
validates actions, and extracts domain-specific safety telemetry.
"""

from collections.abc import Callable
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


_AdapterEntry = type[BaseModalityAdapter] | BaseModalityAdapter | Callable[..., BaseModalityAdapter]
#: plugin-id -> (factory, content digest). The digest is the SHA-256 of the
#: plugin module content, verified at bootstrap from a pinned manifest and
#: served by :func:`adapter_revision` so task stream pins can be checked.
_ADAPTER_REGISTRY: dict[str, tuple[_AdapterEntry, str]] = {}


def register_adapter(
    plugin_id: ModalityKind | str,
    adapter: _AdapterEntry,
    *,
    digest: str = "",
    override: bool = False,
) -> None:
    """Register an adapter class, instance, or factory for a plugin id.

    ``digest`` is the SHA-256 content identity of the plugin implementing the
    adapter; a non-empty digest is what lets a task pin a stream to an exact
    adapter version. When empty, the plugin is marked unpinned (digest `""`).
    """
    key = plugin_id.value if isinstance(plugin_id, ModalityKind) else str(plugin_id)
    if key in _ADAPTER_REGISTRY and not override:
        raise TaskContractError(f"modality adapter already registered for {key!r}")
    _ADAPTER_REGISTRY[key] = (adapter, digest)


def get_adapter(plugin_id: ModalityKind | str, **kwargs: Any) -> BaseModalityAdapter | None:
    """Get an instantiated adapter for a plugin id, or None if not registered."""
    key = plugin_id.value if isinstance(plugin_id, ModalityKind) else str(plugin_id)
    entry = _ADAPTER_REGISTRY.get(key)
    if entry is None:
        return None
    factory, _digest = entry
    if callable(factory):
        return factory(**kwargs)
    return factory


def adapter_revision(plugin_id: ModalityKind | str) -> str:
    """Return the SHA-256 content identity of a registered adapter plugin.

    Returns the empty string for an unpinned (locally-ad-hoc) plugin or when
    the plugin id is unknown. Task load refuses to pin a stream to an
    unknown/unpinned adapter (comparison against ``""`` never matches).
    """
    key = plugin_id.value if isinstance(plugin_id, ModalityKind) else str(plugin_id)
    entry = _ADAPTER_REGISTRY.get(key)
    return "" if entry is None else entry[1]


def require_adapter(plugin_id: ModalityKind | str, **kwargs: Any) -> BaseModalityAdapter:
    """Get an adapter or raise TaskContractError if not registered."""
    adapter = get_adapter(plugin_id, **kwargs)
    if adapter is None:
        key = plugin_id.value if isinstance(plugin_id, ModalityKind) else str(plugin_id)
        known = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
        raise TaskContractError(f"unknown modality {key!r}; registered adapters: {known}")
    return adapter


def list_adapters() -> dict[str, str]:
    """Return dictionary of registered plugin ids and their adapter class names."""
    result = {}
    for k, (v, _digest) in sorted(_ADAPTER_REGISTRY.items()):
        if isinstance(v, type):
            result[k] = v.__name__
        elif callable(v):
            result[k] = getattr(v, "__name__", "factory")
        else:
            result[k] = v.__class__.__name__
    return result


def clear_registry() -> None:
    """Reset the adapter registry (primarily for test isolation)."""
    _ADAPTER_REGISTRY.clear()


def reset_default_adapters() -> None:
    """Reset and re-populate the adapter registry from the bundled manifest."""
    _ADAPTER_REGISTRY.clear()
    from or_audit.eval.adapters.manifest import bootstrap_adapter_plugins

    bootstrap_adapter_plugins()
