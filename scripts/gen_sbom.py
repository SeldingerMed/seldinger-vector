"""SPDX 2.3 SBOM generator for SurgEval from the locked dependency graph.

Produces ``sbom.spdx.json`` at the repo root (or the given ``--out``) by
resolving ``uv.lock`` and ``pyproject.toml``:

- the document ``DESCRIBES`` the project package (``SPDXRef-surgeval``), whose
  version comes from ``pyproject [project]`` (NOT the lock schema version);
  the project's own ``uv.lock`` editable entry is skipped so it is not
  emitted a second time as a locked package;
- every locked *variant* becomes an SPDX package with a unique SPDXID: each
  resolved occurrence of a name (e.g. NumPy built twice for different Python
  markers / versions) is enumerated ``<name>``, ``<name>-1``, ``<name>-2``…;
- license comes from the reviewed table in ``check_license_allowlist``;
  unmapped licenses are ``NOASSERTION`` (the allowlist gate is the authority);
- ``DEPENDS_ON`` relationships model the lock dependency edges from each
  package to its declared dependency name; the root (project) edges derive
  from ``[project].dependencies`` only, excluding optional extras;
- optional fields with no value (e.g. ``supplier`` for git sources) are
  omitted entirely, never emitted as JSON ``null``;
- ``documentNamespace`` is keyed by a SHA-256 of the actual ``uv.lock`` bytes
  and ``creationInfo.created`` is the real UTC timestamp.

Run: ``uv run python scripts/gen_sbom.py [--out sbom.spdx.json]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from check_license_allowlist import THIRD_PARTY  # type: ignore[import-not-found]

SPDX_VERSION = "SPDX-2.3"


def _req_name(req: str) -> str:
    """Extract the package name from a PEP 508 requirement string.

    ``"pydantic>=2.7,<3"`` -> ``"pydantic"``; ``"numpy>=1.26"`` -> ``"numpy"``.
    The name is the leading identifier, cut at an extras/version/whitespace
    delimiter.
    """
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*?)(?:[\[<>=!~ ]|$)", req)
    return match.group(1) if match else req


def _supplier(registry: str | None) -> str | None:
    if not registry:
        return None
    if "nvidia" in registry:
        return "Organization: NVIDIA"
    if "pypi" in registry or "pythonhosted" in registry:
        return "Organization: Python Package Index"
    return None


def _package_entry(name: str, sid: str, version: str, src: str | None) -> dict:
    lic = THIRD_PARTY.get(name, "NOASSERTION")
    entry: dict[str, str] = {
        "name": name,
        "SPDXID": sid,
        "versionInfo": version,
        "licenseConcluded": lic,
        "licenseDeclared": lic,
        "downloadLocation": src or "NOASSERTION",
    }
    supplier = _supplier(src)
    if supplier is not None:
        entry["supplier"] = supplier
    return entry


def _resolve_targets(
    dep: dict, by_version: dict[tuple[str, str], str], by_name: dict[str, list[str]]
) -> list[str]:
    """Resolve a dependency entry to its locked variant SPDXIDs.

    A dependency may pin a specific resolved variant (uv.lock emits
    ``{name, version, marker}`` for marker-split packages such as NumPy built
    once per Python version); match that exactly. Otherwise (name-only edge)
    return every resolved variant of that name so no variant is orphaned.
    """
    name = dep["name"]
    version = dep.get("version")
    if version is not None:
        exact = by_version.get((name, str(version)))
        if exact is not None:
            return [exact]
    return by_name.get(name, [])


def build(lock: Path, pyproject: Path) -> dict:
    with lock.open("rb") as fh:
        lock_bytes = fh.read()
        lock_data = tomllib.loads(lock_bytes.decode())
    with pyproject.open("rb") as fh:
        project = tomllib.load(fh)["project"]

    root_id = "SPDXRef-surgeval"
    packages: list[dict] = [
        {
            "name": "surgeval",
            "SPDXID": root_id,
            "versionInfo": str(project["version"]),
            "supplier": "Organization: SeldingerMed",
            "licenseConcluded": "Apache-2.0",
            "licenseDeclared": "Apache-2.0",
            "copyrightText": "Copyright 2026 SeldingerMed",
            "downloadLocation": "NOASSERTION",
        }
    ]

    # Unique SPDXID per resolved variant: `idx` increments per occurrence of a
    # name regardless of version/markers, so NULL name/version collisions are
    # impossible (e.g. numpy-2.4.6 -> -numpy, numpy-2.5.2 -> -numpy-1).
    # `by_version` maps (name, version) -> SPDXID for version-tagged edges;
    # `by_name` maps name -> [SPDXID, ...] so name-only edges reach every
    # resolved variant of that dependency.
    counters: dict[str, int] = {}
    by_version: dict[tuple[str, str], str] = {}
    by_name: dict[str, list[str]] = {}
    edges_spec: list[tuple[str, dict]] = []

    for pkg in lock_data["package"]:
        # Skip the editable install of the project itself (uv.lock `source =
        # { editable = "." }`); the root package is emitted above as
        # SPDXRef-surgeval with metadata from pyproject.
        if (pkg.get("source") or {}).get("editable"):
            continue
        name = pkg["name"]
        idx = counters.get(name, 0)
        counters[name] = idx + 1
        sid = f"SPDXRef-pkg-{name}" if idx == 0 else f"SPDXRef-pkg-{name}-{idx}"
        version = str(pkg.get("version", ""))
        by_version[(name, version)] = sid
        by_name.setdefault(name, []).append(sid)
        src = (pkg.get("source") or {}).get("registry") or None
        packages.append(_package_entry(name, sid, version, src))
        for dep in pkg.get("dependencies", []):
            edges_spec.append((sid, dep))

    # Root edges come from `[project].dependencies` only (runtime deps).
    # Optional extras (dev, lumen) are intentionally excluded: dev tools are
    # not runtime, and the lumen extra pulls a git-only package with no
    # registry that the allowlist treats as an optional integration.
    for req in project.get("dependencies", []):
        edges_spec.append((root_id, {"name": _req_name(req)}))

    relationships: list[dict] = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": root_id,
        }
    ]
    seen: set[tuple[str, str]] = set()
    for src, dep in edges_spec:
        for target in _resolve_targets(dep, by_version, by_name):
            if target != src and (src, target) not in seen:
                seen.add((src, target))
                relationships.append(
                    {
                        "spdxElementId": src,
                        "relationshipType": "DEPENDS_ON",
                        "relatedSpdxElement": target,
                    }
                )

    return {
        "spdxVersion": SPDX_VERSION,
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "surgeval-sbom",
        "documentNamespace": (
            f"https://seldingermed/surgeval/sbom/{hashlib.sha256(lock_bytes).hexdigest()[:32]}"
        ),
        "creationInfo": {
            "created": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "creators": ["Tool: SurgEval/gen_sbom.py"],
        },
        "packages": packages,
        "relationships": relationships,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sbom.spdx.json")
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    doc = build(root / "uv.lock", root / "pyproject.toml")
    out = root / args.out
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
