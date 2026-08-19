"""Tests for multi-modality contracts, GateKind extensions, and ModalityAdapter."""

from collections.abc import Iterator
from typing import Any

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.adapters.base import (
    BaseModalityAdapter,
    ModalityAdapter,
    clear_registry,
    get_adapter,
    list_adapters,
    register_adapter,
    require_adapter,
    reset_default_adapters,
)
from or_audit.eval.contracts import CapabilitySpec, InteractionMode, InterfaceSpec
from or_audit.eval.enums import GateKind, ModalityKind
from or_audit.eval.task import GateSpec, TaskMetadata


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    reset_default_adapters()
    yield
    reset_default_adapters()


def test_modality_kind_enum_values() -> None:
    assert str(ModalityKind.VIDEO_LAPAROSCOPIC) == "video-laparoscopic"
    assert str(ModalityKind.VIDEO_ENDOSCOPIC) == "video-endoscopic"
    assert str(ModalityKind.AIRWAY_BRONCHOSCOPY) == "airway-bronchoscopy"
    assert str(ModalityKind.FLUOROSCOPY_DSA) == "fluoroscopy-dsa"
    assert str(ModalityKind.ORTHOPEDIC_POINTCLOUD) == "orthopedic-pointcloud"
    assert str(ModalityKind.ROBOTIC_KINEMATICS) == "robotic-kinematics"
    assert str(ModalityKind.ENDOVASCULAR_SIM) == "endovascular-sim"
    assert str(ModalityKind.SYNTHETIC_PROCEDURAL) == "synthetic-procedural"


def test_gate_kind_enum_values() -> None:
    assert str(GateKind.SPATIAL_EXCLUSION) == "spatial-exclusion"
    assert str(GateKind.FORCE_THRESHOLD) == "force-threshold"
    assert str(GateKind.PERFORATION_RISK) == "perforation-risk"
    assert str(GateKind.RADIATION_DOSE) == "radiation-dose"
    assert str(GateKind.TEMPORAL_BOUND) == "temporal-bound"
    assert str(GateKind.CUSTOM) == "custom"


def test_gate_spec_with_kind_and_threshold() -> None:
    gate = GateSpec(
        id="critical_view_safety",
        source="oracle.distance_to_cbd",
        fail_when="distance < 2.0",
        maps_to="cbd_injury_risk",
        kind=GateKind.SPATIAL_EXCLUSION,
        threshold=2.0,
        unit="mm",
    )
    assert gate.kind == GateKind.SPATIAL_EXCLUSION
    assert gate.threshold == 2.0
    assert gate.unit == "mm"


def test_gate_spec_defaults() -> None:
    gate = GateSpec(id="default_gate", source="info.unsafe", fail_when="unsafe == true")
    assert gate.kind == GateKind.CUSTOM
    assert gate.threshold is None
    assert gate.unit == ""


def test_interface_and_capability_modalities() -> None:
    interface = InterfaceSpec(
        id="laparoscopic-action",
        interaction_mode=InteractionMode.CLOSED_LOOP,
        observations=("stereo-rgb",),
        actions=("tool-pose",),
        modalities=(ModalityKind.VIDEO_LAPAROSCOPIC.value,),
    )
    assert interface.modalities == ("video-laparoscopic",)

    matching_cap = CapabilitySpec(
        interface="laparoscopic-action",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        observations=("stereo-rgb",),
        actions=("tool-pose",),
        modalities=(ModalityKind.VIDEO_LAPAROSCOPIC.value,),
    )
    assert matching_cap.satisfies(interface)

    mismatched_cap = CapabilitySpec(
        interface="laparoscopic-action",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        observations=("stereo-rgb",),
        actions=("tool-pose",),
        modalities=(ModalityKind.FLUOROSCOPY_DSA.value,),
    )
    assert not mismatched_cap.satisfies(interface)

    wildcard_cap = CapabilitySpec(
        interface="laparoscopic-action",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        schema_wildcard=True,
    )
    assert wildcard_cap.satisfies(interface)


def test_task_metadata_modality() -> None:
    meta = TaskMetadata(
        title="Cholecystectomy Phase Recognition",
        modality=ModalityKind.VIDEO_LAPAROSCOPIC.value,
        tags=("laparoscopy", "phase"),
    )
    assert meta.modality == "video-laparoscopic"


class DummyBronchoAdapter(ModalityAdapter):
    modality: ModalityKind | str = ModalityKind.AIRWAY_BRONCHOSCOPY

    def validate_observation(self, observation: Any) -> bool:
        return isinstance(observation, dict) and "camera_frame" in observation


def test_adapter_registry() -> None:
    clear_registry()
    assert isinstance(DummyBronchoAdapter(), BaseModalityAdapter)
    register_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY, DummyBronchoAdapter)
    assert "airway-bronchoscopy" in list_adapters()

    adapter = get_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY)
    assert adapter is not None
    assert isinstance(adapter, DummyBronchoAdapter)
    assert adapter.validate_observation({"camera_frame": [1, 2, 3]})
    assert not adapter.validate_observation("invalid")

    # Duplicate registration raises without override
    with pytest.raises(TaskContractError, match="already registered"):
        register_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY, DummyBronchoAdapter)

    # Override works
    register_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY, DummyBronchoAdapter, override=True)

    # Require adapter succeeds when registered
    req = require_adapter(ModalityKind.AIRWAY_BRONCHOSCOPY)
    assert isinstance(req, DummyBronchoAdapter)

    # Require adapter fails when missing
    with pytest.raises(TaskContractError, match="unknown modality"):
        require_adapter(ModalityKind.ORTHOPEDIC_POINTCLOUD)


def test_gate_spec_normalization() -> None:
    # Snake case gets normalized to kebab case enum
    gate = GateSpec(id="force_gate", kind="force_threshold", threshold=1.5, unit="N")
    assert gate.kind == GateKind.FORCE_THRESHOLD

    # Custom string remains custom slug
    custom_gate = GateSpec(id="my_gate", kind="my-custom-kind")
    assert custom_gate.kind == "my-custom-kind"


def test_adapter_safety_extraction_defensive() -> None:
    adapter = ModalityAdapter()
    # None or non-dict input returns empty dict
    assert adapter.extract_safety_state(None) == {}
    assert adapter.extract_safety_state({}) == {}
    # None values inside context are handled gracefully
    assert adapter.extract_safety_state({"info": None, "safety": None}) == {}
    # Extracts known keys from info into safety
    result = adapter.extract_safety_state(
        {"info": {"max_pen": 0.05, "other_metric": 123}, "safety": {"existing": True}}
    )
    assert result == {"existing": True, "max_pen": 0.05}


def test_adapter_instance_registration() -> None:
    instance = DummyBronchoAdapter()
    register_adapter("custom-broncho", instance)
    retrieved = get_adapter("custom-broncho")
    assert retrieved is instance


def test_eval_init_exports() -> None:
    import or_audit.eval as eval_module

    assert hasattr(eval_module, "ModalityKind")
    assert hasattr(eval_module, "GateKind")
    assert hasattr(eval_module, "GateSpec")
    assert hasattr(eval_module, "BaseModalityAdapter")
    assert hasattr(eval_module, "ModalityAdapter")
    assert hasattr(eval_module, "register_adapter")
    assert hasattr(eval_module, "get_adapter")
