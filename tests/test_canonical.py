"""Canonical serialization contract.

These tests defend hash stability. If any of them fail, previously written
audit chains stop verifying.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum

import pytest

from or_audit.audit.canonical import canonical_json, digest


class _Colour(StrEnum):
    RED = "red"


def test_key_order_does_not_affect_output():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_nested_key_order_does_not_affect_output():
    left = {"outer": {"z": [1, {"b": 2, "a": 3}], "y": 4}}
    right = {"outer": {"y": 4, "z": [1, {"a": 3, "b": 2}]}}
    assert digest(left) == digest(right)


def test_list_order_does_affect_output():
    """Sequences are ordered data; reordering them must change the hash."""
    assert digest([1, 2]) != digest([2, 1])


def test_no_insignificant_whitespace():
    assert canonical_json({"a": [1, 2]}) == '{"a":[1,2]}'


def test_equal_instants_in_different_offsets_hash_equally():
    utc = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    plus_two = utc.astimezone(timezone(timedelta(hours=2)))
    assert digest({"t": utc}) == digest({"t": plus_two})


def test_naive_datetime_rejected():
    with pytest.raises(ValueError, match="naive datetime"):
        canonical_json({"t": datetime(2026, 1, 1, 12, 0)})


def test_enum_serializes_as_its_value():
    assert canonical_json({"c": _Colour.RED}) == '{"c":"red"}'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_rejected(bad):
    """NaN is not equal to itself, so a hash containing it means nothing."""
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"x": bad})


def test_non_string_keys_rejected():
    with pytest.raises(TypeError, match="object keys must be strings"):
        canonical_json({1: "a"})


def test_unsupported_type_rejected():
    with pytest.raises(TypeError, match="no canonical JSON form"):
        canonical_json({"x": object()})


def test_unicode_is_not_escaped():
    assert canonical_json({"k": "café"}) == '{"k":"café"}'


def test_digest_is_stable_hex_sha256():
    assert digest({}) == digest({})
    assert len(digest({})) == 64
