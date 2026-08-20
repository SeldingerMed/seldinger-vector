"""Commercial license-allowlist gate for SurgEval runtime dependencies.

SurgEval ships as open-core (Apache-2.0) with commercial/SaaS and
regulatory-attestation tiers, so every *runtime* dependency must carry a
commercial-friendly license. This script gates the resolved runtime closure
from ``uv.lock`` against a curated, reviewed table:

1. Every runtime dependency reachable from ``[project].dependencies`` must be
   present in :data:`THIRD_PARTY` (an explicit maintainer review — a new
   dependency simply failing here is the intended governance mechanism).
2. Every mapped license must be in :data:`ALLOWLIST` (permissive, sublicensable,
   no copyleft/commodity restrictions). MPL-2.0 is *not* in the runtime
   allowlist (file-level copyleft); it may appear only as a dev/build
   dependency, which this script does not gate.

Run from the repo root: ``uv run python scripts/check_license_allowlist.py``
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

#: name -> SPDX license expression. A dependency's *absence here is a hard
#: failure*: adding a runtime dependency without an explicit reviewed entry
#: is rejected, forcing a deliberate review in review/PR.
THIRD_PARTY: dict[str, str] = {
    "pydantic": "MIT",
    "pydantic-core": "MIT",
    "annotated-types": "MIT",
    "typing-extensions": "PSF-2.0",
    "typing-inspection": "MIT",
    "numpy": "BSD-3-Clause AND MIT AND 0BSD AND Zlib AND CC0-1.0",
    "cloudpickle": "BSD-3-Clause",
}

#: Licenses acceptable for commercial redistribution (permissive, sublicensable).
ALLOWLIST: frozenset[str] = frozenset(
    {
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "Apache-2.0",
        "PSF-2.0",
        "0BSD",
        "Zlib",
        "ISC",
        "CC0-1.0",
    }
)

#: Package names that must not appear in the runtime allowlist table; these are
#: not part of the shipped runtime and are handled separately.
REPO_PACKAGE = "surgeval"


def _prod_dependencies(pyproject: Path) -> list[str]:
    """Top-level runtime dependency names from pyproject, extras stripped."""
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    names: list[str] = []
    for dep in data["project"]["dependencies"]:
        name = dep.split(";", 1)[0].strip()
        name = name.split("[", 1)[0].strip()
        # strip version specifier / extras / markers
        name = name.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].split("~=", 1)[0]
        names.append(name.strip())
    return names


def _resolve_closure(lock: Path, roots: list[str]) -> set[str]:
    """BFS over the lock graph from the given roots (conservative superset)."""
    with lock.open("rb") as fh:
        data = tomllib.load(fh)
    graph: dict[str, list[str]] = {}
    for pkg in data["package"]:
        graph[pkg["name"]] = [d["name"] for d in pkg.get("dependencies", [])]
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        for dep in graph.get(name, []):
            if dep not in seen:
                stack.append(dep)
    return seen


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    lock = root / "uv.lock"
    prods = _prod_dependencies(pyproject)
    closure = _resolve_closure(lock, prods) - {REPO_PACKAGE}

    errors: list[str] = []

    missing_review = sorted(name for name in closure if name not in THIRD_PARTY)
    if missing_review:
        errors.append(
            "runtime dependencies with no reviewed license entry: " + ", ".join(missing_review)
        )

    not_allowed = sorted(
        name for name, lic in THIRD_PARTY.items() if name in closure and not _allowed(lic)
    )
    if not_allowed:
        how = [f"{n} ({THIRD_PARTY[n]})" for n in not_allowed]
        errors.append("runtime dependencies outside the commercial allowlist: " + ", ".join(how))

    unmapped_roots = sorted(n for n in prods if n not in THIRD_PARTY)
    if unmapped_roots:
        errors.append(
            "top-level project dependencies missing from table: " + ", ".join(unmapped_roots)
        )

    if errors:
        print("FAIL: SurgEval licensing gate", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 1

    print(f"OK: {len(closure)} runtime dependencies within commercial allowlist")
    return 0


def _allowed(expression: str) -> bool:
    """True if every SPDX term in an AND-composed expression is allowlisted."""
    for term in expression.split(" AND "):
        term = term.strip()
        if term not in ALLOWLIST:
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
