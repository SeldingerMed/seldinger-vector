"""Regression tests for the SPDX SBOM generator (scripts/gen_sbom.py).

Guards the two failure modes the generator must avoid:
- the project's own ``uv.lock`` editable entry must not be re-emitted as a
  second SPDX package (duplicate project SPDXID / name);
- root ``DEPENDS_ON`` edges must derive from ``[project].dependencies`` only
  (runtime), not from every locked name including dev/transitive packages.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_sbom  # type: ignore[import-not-found]  # noqa: E402

LOCK_TMPL = """\
version = 1
requires-python = ">=3.11"

[[package]]
name = "surgeval"
version = "0.3.0a0"
source = { editable = "." }
dependencies = [
    { name = "runtime-a" },
    { name = "runtime-b", version = "2.0", marker = "python_full_version < '3.12'" },
    { name = "runtime-b", version = "2.1", marker = "python_full_version >= '3.12'" },
]

[[package]]
name = "runtime-a"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
dependencies = []

[[package]]
name = "runtime-b"
version = "2.0"
source = { registry = "https://pypi.org/simple" }
dependencies = []

[[package]]
name = "runtime-b"
version = "2.1"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "runtime-a" },
]

[[package]]
name = "consumer"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "runtime-b", version = "2.1" },
]

[[package]]
name = "dev-tool"
version = "3.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "runtime-a" },
]
"""

PYPROJECT_TMPL = """\
[project]
name = "surgeval"
version = "0.3.0a0"
dependencies = ["runtime-a>=1.0", "runtime-b>=2.0"]

[project.optional-dependencies]
dev = ["dev-tool>=3.0"]
"""


def _build(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "uv.lock").write_text(LOCK_TMPL)
    (tmp_path / "pyproject.toml").write_text(PYPROJECT_TMPL)
    # gen_sbom.build is weakly typed (returns untyped dict).
    return gen_sbom.build(  # type: ignore[no-any-return]
        tmp_path / "uv.lock",
        tmp_path / "pyproject.toml",
    )


def _root_depends_on_sids(doc: dict[str, Any]) -> set[str]:
    root = "SPDXRef-surgeval"
    return {
        r["relatedSpdxElement"]
        for r in doc["relationships"]
        if r["spdxElementId"] == root and r["relationshipType"] == "DEPENDS_ON"
    }


def _root_depends_on_names(doc: dict[str, Any]) -> set[str]:
    related = _root_depends_on_sids(doc)
    return {p["name"] for p in doc["packages"] if p["SPDXID"] in related}


def _depends_on_sids(doc: dict[str, Any], src_sid: str) -> set[str]:
    return {
        r["relatedSpdxElement"]
        for r in doc["relationships"]
        if r["spdxElementId"] == src_sid and r["relationshipType"] == "DEPENDS_ON"
    }


def test_skips_editable_project_entry(tmp_path: Path) -> None:
    doc = _build(tmp_path)
    ids = [p["SPDXID"] for p in doc["packages"]]
    names = [p["name"] for p in doc["packages"]]
    # Exactly one project package, never a re-emitted locked `SPDXRef-pkg-*`.
    assert ids.count("SPDXRef-surgeval") == 1
    assert names.count("surgeval") == 1
    assert "SPDXRef-pkg-surgeval" not in ids
    # The third-party packages still appear, including both runtime-b variants.
    for sid in (
        "SPDXRef-pkg-runtime-a",
        "SPDXRef-pkg-runtime-b",
        "SPDXRef-pkg-runtime-b-1",
        "SPDXRef-pkg-consumer",
        "SPDXRef-pkg-dev-tool",
    ):
        assert sid in ids
    # Document DESCRIBES the project package.
    describes = [
        r["relatedSpdxElement"]
        for r in doc["relationships"]
        if r["spdxElementId"] == "SPDXRef-DOCUMENT" and r["relationshipType"] == "DESCRIBES"
    ]
    assert describes == ["SPDXRef-surgeval"]


def test_root_links_all_marker_variants(tmp_path: Path) -> None:
    """A name-only root dependency must reach every resolved variant (the
    marker-split case, e.g. NumPy once per Python version), not just the
    first — no variant may be left with zero inbound edges."""
    doc = _build(tmp_path)
    root_deps = _root_depends_on_sids(doc)
    assert "SPDXRef-pkg-runtime-b" in root_deps
    assert "SPDXRef-pkg-runtime-b-1" in root_deps


def test_version_tagged_dependency_resolves_to_exact_variant(tmp_path: Path) -> None:
    """A dependency edge pinned by version (uv.lock marker/keyed entries) must
    resolve to its exact variant, not the first of the name."""
    doc = _build(tmp_path)
    consumer_deps = _depends_on_sids(doc, "SPDXRef-pkg-consumer")
    assert consumer_deps == {"SPDXRef-pkg-runtime-b-1"}


def test_root_edges_from_project_dependencies_only(tmp_path: Path) -> None:
    doc = _build(tmp_path)
    root_deps = _root_depends_on_names(doc)
    assert root_deps == {"runtime-a", "runtime-b"}
    # The dev-only extra and its transitive dep must NOT appear as root edges.
    assert "dev-tool" not in root_deps


def test_real_lock_has_no_duplicate_project() -> None:
    doc = gen_sbom.build(ROOT / "uv.lock", ROOT / "pyproject.toml")
    ids = [p["SPDXID"] for p in doc["packages"]]
    assert len(ids) == len(set(ids)), "duplicate SPDXIDs in generated SBOM"
    assert "SPDXRef-pkg-surgeval" not in ids
    assert ids.count("SPDXRef-surgeval") == 1
    # Real runtime deps from [project].dependencies.
    root_deps = _root_depends_on_names(doc)
    assert root_deps == {"cloudpickle", "numpy", "pydantic"}
    # Marker-split numpy must not be orphaned: both variants are linked.
    root_sids = _root_depends_on_sids(doc)
    assert "SPDXRef-pkg-numpy" in root_sids
    assert "SPDXRef-pkg-numpy-1" in root_sids
