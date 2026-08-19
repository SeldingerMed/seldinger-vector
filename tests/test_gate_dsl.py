"""Conformance tests for the declarative gate DSL evaluator."""

from __future__ import annotations

from or_audit.domain.enums import GateStatus
from or_audit.eval.gate_dsl import evaluate_gate
from or_audit.eval.task import GateSpec


def _gate(
    gate_id: str = "g",
    source: str = "contact_force",
    fail_when: str = "contact_force > 1.5",
    kind: str = "force-threshold",
) -> GateSpec:
    return GateSpec(id=gate_id, source=source, fail_when=fail_when, kind=kind)


def test_gate_passes_when_below_threshold() -> None:
    gate = _gate()
    outcome = evaluate_gate(gate, {"contact_force": 1.2})
    assert outcome is not None
    assert outcome.status is GateStatus.PASS


def test_gate_fails_when_above_threshold() -> None:
    gate = _gate()
    outcome = evaluate_gate(gate, {"contact_force": 2.0})
    assert outcome is not None
    assert outcome.status is GateStatus.FAIL


def test_gate_not_assessable_when_signal_missing() -> None:
    gate = _gate()
    outcome = evaluate_gate(gate, {})
    assert outcome is not None
    assert outcome.status is GateStatus.NOT_ASSESSABLE


def test_gate_resolves_dotted_source_to_short_name() -> None:
    gate = _gate(source="oracle.catheter.contact_force", fail_when="contact_force > 1.5")
    outcome = evaluate_gate(gate, {"contact_force": 2.0})
    assert outcome is not None
    assert outcome.status is GateStatus.FAIL


def test_gate_resolves_dotted_source_to_full_path() -> None:
    gate = _gate(source="oracle.catheter.contact_force", fail_when="contact_force > 1.5")
    outcome = evaluate_gate(gate, {"oracle.catheter.contact_force": 2.0})
    assert outcome is not None
    assert outcome.status is GateStatus.FAIL


def test_gate_boolean_equality() -> None:
    gate = _gate(
        source="cbd_violation", fail_when="cbd_violation == true", kind="spatial-exclusion"
    )
    fail = evaluate_gate(gate, {"cbd_violation": True})
    assert fail is not None
    assert fail.status is GateStatus.FAIL
    passing = evaluate_gate(gate, {"cbd_violation": False})
    assert passing is not None
    assert passing.status is GateStatus.PASS


def test_gate_with_threshold_field() -> None:
    gate = GateSpec(
        id="overshoot",
        source="max_overshoot_mm",
        fail_when="max_overshoot_mm > threshold",
        kind="spatial-exclusion",
        threshold=0.5,
    )
    outcome = evaluate_gate(gate, {"max_overshoot_mm": 0.7, "threshold": 0.5})
    assert outcome is not None
    assert outcome.status is GateStatus.FAIL


def test_gate_compound_expression() -> None:
    gate = _gate(
        source="force",
        fail_when="force > 1.5 and speed > 10",
        kind="force-threshold",
    )
    fail = evaluate_gate(gate, {"force": 2.0, "speed": 15})
    assert fail is not None
    assert fail.status is GateStatus.FAIL
    passing = evaluate_gate(gate, {"force": 2.0, "speed": 5})
    assert passing is not None
    assert passing.status is GateStatus.PASS


def test_gate_returns_none_when_no_declarative_spec() -> None:
    gate = GateSpec(id="manual_gate")
    assert evaluate_gate(gate, {"x": 1}) is None


def test_conformance_two_tasks_same_kind_compute_identically() -> None:
    """Two tasks declaring force_threshold must compute the same outcome from the same signal."""
    gate_a = _gate(gate_id="airway_wall", source="contact_force", fail_when="contact_force > 1.5")
    gate_b = _gate(gate_id="vessel_wall", source="contact_force", fail_when="contact_force > 1.5")
    for value in [0.5, 1.5, 2.0, 3.0]:
        out_a = evaluate_gate(gate_a, {"contact_force": value})
        out_b = evaluate_gate(gate_b, {"contact_force": value})
        assert out_a is not None
        assert out_b is not None
        assert out_a.status is out_b.status, f"mismatch at contact_force={value}"


def test_gate_malformed_expression_is_not_assessable() -> None:
    gate = _gate(fail_when="contact_force >>> 1.5")
    outcome = evaluate_gate(gate, {"contact_force": 2.0})
    assert outcome is not None
    assert outcome.status is GateStatus.NOT_ASSESSABLE
