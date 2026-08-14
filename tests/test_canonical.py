"""Canonical serialization contract.

These tests defend hash stability across time and across implementations. If
any of them fail, previously written audit chains stop verifying, so the
golden vectors below are deliberately hardcoded rather than derived — a
self-consistency assertion (``A == A``) would pass straight through a format
drift, which is exactly the failure worth catching.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum

import pytest

from or_audit.audit.canonical import MAX_SAFE_INTEGER, canonical_json, digest, digest_text


class _Colour(StrEnum):
    RED = "red"


class TestGoldenVectors:
    """Frozen input/output pairs. Changing these breaks existing chains."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ({}, "{}"),
            ([], "[]"),
            ({"a": 1}, '{"a":1}'),
            ({"b": 1, "a": 2}, '{"a":2,"b":1}'),
            ({"n": None, "t": True, "f": False}, '{"f":false,"n":null,"t":true}'),
            ({"s": "café"}, '{"s":"café"}'),
            ({"s": 'quote"and\\slash'}, '{"s":"quote\\"and\\\\slash"}'),
            ({"s": "tab\tnewline\n"}, '{"s":"tab\\tnewline\\n"}'),
            ({"s": "\x00\x1f"}, '{"s":"\\u0000\\u001f"}'),
            ({"s": "\b\f\r"}, '{"s":"\\b\\f\\r"}'),
            (
                {"nested": {"z": [1, {"b": 2, "a": 3}], "y": 4}},
                '{"nested":{"y":4,"z":[1,{"a":3,"b":2}]}}',
            ),
            (
                {"t": datetime(2026, 3, 4, 14, 30, tzinfo=UTC)},
                '{"t":"2026-03-04T14:30:00.000000Z"}',
            ),
        ],
    )
    def test_canonical_form(self, value, expected):
        assert canonical_json(value) == expected

    def test_digest_golden(self):
        """Pin one full digest so a hashing change cannot slip through."""
        assert digest({"a": 1}) == digest_text('{"a":1}')
        assert digest({}) == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


class TestNumberCanon:
    """RFC 8785 / ECMAScript number rules, not Python's repr."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0"),
            (0.0, "0"),
            (-0.0, "0"),
            (1, "1"),
            (1.0, "1"),
            (-1.0, "-1"),
            (1.5, "1.5"),
            (100.0, "100"),
            (0.001, "0.001"),
            (1e-7, "1e-7"),
            (1e21, "1e+21"),
            (1e20, "100000000000000000000"),
            (1e16, "10000000000000000"),
            (1.25e-10, "1.25e-10"),
            (MAX_SAFE_INTEGER, "9007199254740991"),
        ],
    )
    def test_number_rendering(self, value, expected):
        """Python renders 1.0 as '1.0' and 1e16 as '1e+16'; JCS does not."""
        assert canonical_json(value) == expected

    def test_integral_float_and_int_render_identically(self):
        """JSON has one number type; 1 and 1.0 must not hash differently."""
        assert digest({"x": 1}) == digest({"x": 1.0})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_rejected(self, bad):
        """NaN is not equal to itself, so a hash containing it means nothing."""
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json({"x": bad})

    @pytest.mark.parametrize("bad", [MAX_SAFE_INTEGER + 1, -(MAX_SAFE_INTEGER + 1), 2**70])
    def test_unsafe_integers_rejected(self, bad):
        """An integer a conforming JSON parser cannot round-trip has no canon."""
        with pytest.raises(ValueError, match="exact IEEE-754 double range"):
            canonical_json({"x": bad})


class TestStructuralRules:
    def test_key_order_does_not_affect_output(self):
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_nested_key_order_does_not_affect_output(self):
        left = {"outer": {"z": [1, {"b": 2, "a": 3}], "y": 4}}
        right = {"outer": {"y": 4, "z": [1, {"a": 3, "b": 2}]}}
        assert digest(left) == digest(right)

    def test_list_order_does_affect_output(self):
        """Sequences are ordered data; reordering them must change the hash."""
        assert digest([1, 2]) != digest([2, 1])

    def test_keys_sort_by_utf16_code_unit_not_code_point(self):
        """RFC 8785 orders keys by UTF-16 code unit.

        U+10000 encodes as the surrogate pair D800 DC00, so UTF-16 places it
        before U+E000..U+FFFF, while Python's default code-point sort places
        it after. Getting this wrong silently produces digests no conforming
        implementation can reproduce.
        """
        rendered = canonical_json(dict.fromkeys(["\uff21", "\U00010000", "a", "\ue000"], 1))
        assert rendered == '{"a":1,"\U00010000":1,"\ue000":1,"\uff21":1}'

    def test_key_order_is_independent_of_insertion_order(self):
        keys = ["\uff21", "\U00010000", "a", "\ue000"]
        forward = canonical_json(dict.fromkeys(keys, 1))
        backward = canonical_json(dict.fromkeys(reversed(keys), 1))
        assert forward == backward

    def test_supplementary_characters_are_not_escaped(self):
        assert canonical_json({"k": "\U0001f600"}) == '{"k":"\U0001f600"}'

    def test_enum_serializes_as_its_value(self):
        assert canonical_json({"c": _Colour.RED}) == '{"c":"red"}'

    def test_non_string_keys_rejected(self):
        with pytest.raises(TypeError, match="object keys must be strings"):
            canonical_json({1: "a"})

    def test_unsupported_type_rejected(self):
        with pytest.raises(TypeError, match="no canonical JSON form"):
            canonical_json({"x": object()})


class TestDatetimeRules:
    def test_equal_instants_in_different_offsets_hash_equally(self):
        utc = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        plus_two = utc.astimezone(timezone(timedelta(hours=2)))
        assert digest({"t": utc}) == digest({"t": plus_two})

    def test_naive_datetime_rejected(self):
        with pytest.raises(ValueError, match="naive datetime"):
            canonical_json({"t": datetime(2026, 1, 1, 12, 0)})

    def test_microseconds_preserved(self):
        moment = datetime(2026, 1, 1, 12, 0, 0, 123456, tzinfo=UTC)
        assert canonical_json({"t": moment}) == '{"t":"2026-01-01T12:00:00.123456Z"}'


def test_digest_is_stable_hex_sha256():
    assert len(digest({})) == 64
    assert digest({}) == digest({})
