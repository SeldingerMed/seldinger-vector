"""Differential conformance of the number canon against real ECMAScript.

PLAN.md sections 7.3 and 9 contemplate the audit chain being re-verified by a
third party. That claim is only true if our canonical form matches the spec,
and RFC 8785 defines number serialization by reference to ECMAScript
``Number::toString``. Hand-written unit vectors cannot cover the shortest-
round-trip and fixed-versus-exponential boundary logic; a differential test
against a real ECMAScript engine can.

Skipped when Node is unavailable, so a contributor without it still gets a
usable local run. CI runners have Node preinstalled, so this executes there.
"""

from __future__ import annotations

import json
import random
import shutil
import struct
import subprocess
import sys

import pytest

from or_audit.audit.canonical import MAX_SAFE_INTEGER, _format_number

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="requires Node for ECMAScript reference output"
)

# Bit patterns exceed 2**53, so they must cross the process boundary as
# strings. Passing them as JSON numbers silently rounds them and produces
# bogus "divergences" against values that were never tested.
_REFERENCE_JS = """
const bits = JSON.parse(require('fs').readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(bits.map(b => {
  const buf = Buffer.alloc(8);
  buf.writeBigUInt64BE(BigInt(b));
  return buf.readDoubleBE(0).toString();
})));
"""


def _ecmascript_render(values: list[float]) -> list[str]:
    """Render each double exactly as ECMAScript ``Number::toString`` would."""
    payload = json.dumps([str(struct.unpack(">Q", struct.pack(">d", v))[0]) for v in values])
    result = subprocess.run(
        ["node", "-e", _REFERENCE_JS],
        input=payload.encode(),
        capture_output=True,
        check=True,
    )
    rendered: list[str] = json.loads(result.stdout)
    return rendered


def _candidate_doubles() -> list[float]:
    """Random bit patterns plus values near every interesting boundary."""
    rng = random.Random(20260304)
    values: list[float] = []
    for _ in range(20000):
        candidate = struct.unpack(">d", struct.pack(">Q", rng.getrandbits(64)))[0]
        if candidate == candidate and candidate not in (float("inf"), float("-inf")):
            values.append(candidate)

    values += [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.5,
        -1.5,
        0.1,
        0.2,
        0.3,
        1 / 3,
        2 / 3,
        # Fixed/exponential switch points in both directions.
        1e-7,
        1e-6,
        1e-5,
        1e15,
        1e16,
        1e17,
        1e20,
        1e21,
        1e22,
        0.000001,
        0.0000001,
        1.1e21,
        9.999999999999999e20,
        # Precision and range extremes.
        5e-324,
        4.9e-324,
        2.2250738585072014e-308,
        1.7976931348623157e308,
        9007199254740992.0,
        float(MAX_SAFE_INTEGER),
        123456789.123456789,
        123456789012345680000.0,
        1.25e-10,
        2.5e-7,
    ]
    for base in (1e-8, 1e-7, 1e-6, 1e-5, 1e19, 1e20, 1e21, 1e22):
        values.extend(base * m for m in (1.0, 1.01, 1.1, 1.5, 5.0, 9.9, 9.99))

    return list(dict.fromkeys(values))


def test_number_canon_matches_ecmascript():
    """Every finite double must render exactly as ECMAScript renders it."""
    values = _candidate_doubles()
    expected = _ecmascript_render(values)

    divergences = [
        (value, _format_number(value), reference)
        for value, reference in zip(values, expected, strict=True)
        if _format_number(value) != reference
    ]

    assert not divergences, (
        f"{len(divergences)} of {len(values)} doubles diverge from ECMAScript; "
        f"first few: {divergences[:5]}"
    )


def test_reference_harness_would_catch_a_real_divergence():
    """Guard the guard.

    A differential test that cannot fail is worse than none, because it reads
    as evidence. Confirm the reference genuinely disagrees with Python's own
    repr, which is what the implementation deliberately does not use.
    """
    values = [1.0, 1e16, 1e21, 100.0]
    reference = _ecmascript_render(values)
    assert reference == ["1", "10000000000000000", "1e+21", "100"]
    assert [repr(v) for v in values] != reference


def test_python_version_is_recorded_for_triage(capsys):
    """Shortest-repr is stable across CPython versions, but pin the evidence."""
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    assert "python=" in capsys.readouterr().out
