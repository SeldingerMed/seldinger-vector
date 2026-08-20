"""Identity-consolidation gate for the SurgEval distribution.

SurgEval has TWO importable top-level packages in one wheel: the internal
implementation (``or_audit``) and the public SDK shim (``surgeval``). The
public identity is ``surgeval``; ``or_audit`` is an implementation detail and
must never diverge in version, and the wheel must always contain both. This
script enforces that single-sourcing contract so release tooling can rely on a
single version string.

Checks:
1. ``pyproject.toml [project] version`` == ``or_audit.version.PACKAGE_VERSION``
   == ``surgeval.__version__`` (single source of truth).
2. Building the wheel yields exactly one ``or_audit/`` tree and one
   ``surgeval/`` tree (no dropout where the SDK shim is missing a backend).
3. When ``--expected-version`` is given (the release tag/input), the single
   version must equal it — a pre-publication gate so a tag never publishes a
   wheel whose metadata says a different version.

Run from the repo root: ``uv run python scripts/check_identity.py [--expected-version V]``
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-version", default=None)
    args, _ = ap.parse_known_args()
    expected = args.expected_version

    root = Path(__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    errors: list[str] = []

    with pyproject.open("rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]

    import or_audit.version as v

    package_version = v.PACKAGE_VERSION
    try:
        import surgeval as sdk

        sdk_version = sdk.__version__
    except ImportError as exc:  # pragma: no cover - sdk always importable in-repo
        errors.append(f"surgeval SDK not importable: {exc}")
        sdk_version = "<missing>"

    versions = {"pyproject": declared, "or_audit": package_version, "surgeval": sdk_version}
    if len(set(versions.values())) != 1:
        errors.append(f"version mismatch across identity sources: {versions}")
    if expected is not None and package_version != expected:
        errors.append(
            f"release version mismatch: package is {package_version}, expected {expected}"
        )
    # Wheel must contain both trees.
    dist = subprocess.run(
        ["uv", "build", "--out-dir", str(root / ".dist-smoke")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    wheel = next((root / ".dist-smoke").glob("*.whl"), None)
    if wheel is None:
        errors.append(f"wheel build failed: {dist.stderr[-500:]}")
    else:
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
        for pkg in ("or_audit", "surgeval"):
            if not any(n.startswith(f"{pkg}/") for n in names):
                errors.append(f"wheel missing package tree: {pkg}")
        if sum(1 for n in names if n.startswith("or_audit/")) == 0:
            errors.append("wheel has an empty or_audit tree")

    if errors:
        print("FAIL: SurgEval identity gate", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 1

    print(
        f"OK: single identity surgeval==or_audit=={package_version}, "
        f"wheel {wheel.name} ships or_audit/ + surgeval/"
    )
    return 0


if __name__ == "__main__":
    # destroy the .dist-smoke scratch dir we may have just created is handled by
    # the caller (CI cleans the workspace); local runs tolerate it.
    raise SystemExit(_main())
