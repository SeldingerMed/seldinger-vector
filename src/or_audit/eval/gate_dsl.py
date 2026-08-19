"""Declarative gate evaluation from source signals and fail_when expressions.

The kernel evaluates gates declaratively when a task-owned verifier emits
``signals`` alongside ``gates`` and ``metrics``.  Each signal is a named
scalar (bool, int, float, str) that the verifier extracted from oracle
evidence.  For a gate with non-empty ``source`` and ``fail_when``, the
kernel resolves the signal value and evaluates the expression safely,
producing a ``GateOutcome`` without trusting the verifier's own status.

This makes gate semantics comparable across tasks: two tasks declaring
``kind = "force_threshold"`` with ``fail_when = "contact_force > 1.5"``
compute the same outcome from the same signal, regardless of how each
verifier is written.
"""

from __future__ import annotations

import ast
import operator as op
from typing import Any

from or_audit.domain.enums import GateStatus
from or_audit.eval.task import GateSpec
from or_audit.eval.vector import GateOutcome

_OPS: dict[type[ast.AST], Any] = {
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
    ast.And: None,
    ast.Or: None,
    ast.Not: None,
    ast.UnaryOp: None,
}


def _eval_node(node: ast.AST, signals: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, signals)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "true":
            return True
        if node.id == "false":
            return False
        if node.id == "null" or node.id == "none":
            return None
        return signals.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_node(node.operand, signals)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand, signals)
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, signals) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        return any(values)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, signals)
        for comparator, comparator_node in zip(node.ops, node.comparators, strict=True):
            right = _eval_node(comparator_node, signals)
            check = _OPS.get(type(comparator))
            if check is None:
                raise ValueError(f"unsupported operator {type(comparator).__name__}")
            if not check(left, right):
                return False
            left = right
        return True
    raise ValueError(f"unsupported expression node {type(node).__name__}")


def _resolve_signal(gate: GateSpec, signals: dict[str, Any]) -> Any:
    """Resolve a gate's source path to a signal value."""
    source = gate.source
    if source in signals:
        return signals[source]
    short = source.rsplit(".", 1)[-1] if "." in source else source
    return signals.get(short)


def evaluate_gate(gate: GateSpec, signals: dict[str, Any]) -> GateOutcome | None:
    """Evaluate a gate declaratively from signals.

    Returns a GateOutcome, or None if the gate cannot be evaluated
    declaratively (missing source/fail_when or signal absent).
    """
    if not gate.fail_when or not gate.source:
        return None
    value = _resolve_signal(gate, signals)
    if value is None:
        return GateOutcome(
            id=gate.id,
            status=GateStatus.NOT_ASSESSABLE,
            reason=f"signal {gate.source!r} not emitted by verifier",
        )
    enriched = dict(signals)
    short = gate.source.rsplit(".", 1)[-1] if "." in gate.source else gate.source
    enriched.setdefault(short, value)
    enriched.setdefault(gate.source, value)
    try:
        tree = ast.parse(gate.fail_when, mode="eval")
        failed = bool(_eval_node(tree, enriched))
    except Exception as exc:
        return GateOutcome(
            id=gate.id,
            status=GateStatus.NOT_ASSESSABLE,
            reason=f"fail_when expression error: {exc}",
        )
    if failed:
        return GateOutcome(
            id=gate.id,
            status=GateStatus.FAIL,
            reason=f"{gate.fail_when} (source {gate.source} = {value!r})",
        )
    return GateOutcome(
        id=gate.id,
        status=GateStatus.PASS,
        reason=f"not({gate.fail_when}) (source {gate.source} = {value!r})",
    )
