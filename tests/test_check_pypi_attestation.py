"""Regression tests for scripts/check_pypi_attestation.py.

Guards the distribution-set contract: every built wheel/sdist must be served
by PyPI with the exact built digest and a PEP 740 attestation bundle — a
missing expected artifact, an extra served artifact, or a digest mismatch all
fail even when the other files individually check out.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_pypi_attestation as cpa  # type: ignore[import-not-found]  # noqa: E402

WHEEL = "surgeval-0.4.0-py3-none-any.whl"
SDIST = "surgeval-0.4.0.tar.gz"
SHA = "a" * 64


def _built(wheel: bool = True, sdist: bool = True) -> dict[str, str]:
    expected = {}
    if wheel:
        expected[WHEEL] = SHA
    if sdist:
        expected[SDIST] = "b" * 64
    return expected


def _serve(expected: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {"filename": name, "digests": {"sha256": digest}}
        for name, digest in sorted(expected.items())
    ]


class _FakeOpen:
    """Stands in for `_open`; serves a canned index/integrity response."""

    def __init__(self, urls: list[dict[str, Any]], repo: str | None = None) -> None:
        self._urls = urls
        self._repo = repo

    def __call__(self, url: str, accept: str | None = None) -> dict[str, Any]:
        if "/integrity/" in url:
            bundles = [{"publisher": {"repository": self._repo}}] if self._repo else []
            return {"attestation_bundles": bundles}
        return {"urls": self._urls}


rogue_served = [
    *_serve(_built()),
    {"filename": "rogue.whl", "digests": {"sha256": "c" * 64}},
]
mismatch_served = [
    {"filename": WHEEL, "digests": {"sha256": "c" * 64}},
    {"filename": SDIST, "digests": {"sha256": "b" * 64}},
]
extra_served = _serve({WHEEL: SHA})


@pytest.mark.parametrize(
    ("built", "served", "repo", "expect_ok"),
    [
        # exact match, bundle published by repo -> pass
        (_built(), _serve(_built()), "SeldingerMed/seldinger-vector", True),
        # expected sdist missing from the index -> fail
        (_built(), extra_served, "SeldingerMed/seldinger-vector", False),
        # extra served file not in the built set -> fail
        (_built(), rogue_served, "SeldingerMed/seldinger-vector", False),
        # digest mismatch -> fail
        (_built(), mismatch_served, "SeldingerMed/seldinger-vector", False),
        # no PEP 740 bundle -> fail
        (_built(), _serve(_built()), None, False),
    ],
)
def test_check_distribution_set(built, served, repo, expect_ok, monkeypatch) -> None:
    monkeypatch.setattr(cpa, "_open", _FakeOpen(served, repo))
    rc = cpa.check("0.4.0", built, repo)
    assert (rc == 0) is expect_ok


def test_load_expected_parses_sha256sum() -> None:
    dig = "\n".join(
        [
            f"{SHA}  dist/surgeval-0.4.0-py3-none-any.whl",
            "b" * 64 + "  dist/surgeval-0.4.0.tar.gz",
            "c" * 64 + "  sbom.spdx.json",
        ]
    )
    p = Path(__file__).parent / "_digest_smoke.txt"
    p.write_text(dig + "\n")
    try:
        parsed = cpa._load_expected(p)
    finally:
        p.unlink()
    assert parsed[WHEEL] == SHA
    assert parsed[SDIST] == "b" * 64
    assert "sbom.spdx.json" in parsed
