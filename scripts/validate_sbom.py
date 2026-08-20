"""Validate a SurgEval SBOM against the SPDX 2.3 JSON schema.

Preferred path uses ``jsonschema`` against the official SPDX 2.3 JSON schema
(the workflow supplies it via ``--schema``). When no schema file is given or
``jsonschema`` is unavailable, falls back to a structural self-check that
covers the failure modes the full schema guards against: required top-level
fields, unique SPDXIDs, non-null package fields, and well-formed
``creationInfo.created`` datetimes.

Run: ``uv run python scripts/validate_sbom.py [--sbom sbom.spdx.json] [--schema spdx-schema.json]``
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REQUIRED_DOC = {
    "spdxVersion",
    "dataLicense",
    "SPDXID",
    "name",
    "documentNamespace",
    "creationInfo",
    "packages",
    "relationships",
}
REQUIRED_PACKAGE = {
    "name",
    "SPDXID",
    "versionInfo",
    "licenseConcluded",
    "licenseDeclared",
    "downloadLocation",
}


def _structural(doc: dict) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_DOC - set(doc)
    if missing:
        errors.append(f"document missing required keys: {sorted(missing)}")
    if doc.get("spdxVersion") != "SPDX-2.3":
        errors.append(f"spdxVersion expected SPDX-2.3, got {doc.get('spdxVersion')!r}")
    packages = doc.get("packages", [])
    ids = [p.get("SPDXID") for p in packages]
    if len(ids) != len(set(ids)):
        dup = {i for i in ids if ids.count(i) > 1}
        errors.append(f"duplicate SPDXIDs: {sorted(dup)}")
    for p in packages:
        pm = REQUIRED_PACKAGE - set(p)
        if pm:
            errors.append(f"{p.get('SPDXID')}: missing {sorted(pm)}")
        if any(v is None for v in p.values()):
            errors.append(f"{p.get('SPDXID')}: contains JSON null field")
    created = (doc.get("creationInfo") or {}).get("created")
    if created:
        try:
            datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"creationInfo.created not an ISO datetime: {created!r}")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbom", default="sbom.spdx.json")
    ap.add_argument("--schema", default=None)
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    doc = json.loads((root / args.sbom).read_text())

    if args.schema:
        try:
            from jsonschema import Draft7Validator
        except ImportError:
            print("jsonschema not installed; falling back to structural check", file=sys.stderr)
        else:
            schema = json.loads((root / args.schema).read_text())
            errors = sorted(Draft7Validator(schema).iter_errors(doc), key=lambda e: list(e.path))
            if errors:
                for e in errors:
                    print(f"  schema violation: {'/'.join(str(x) for x in e.path)} {e.message}")
                return 1
            print("OK: SBOM valid against SPDX 2.3 schema")
            return 0

    errors = _structural(doc)
    if errors:
        print("FAIL: SBOM structural validation", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 1
    print("OK: SBOM passes structural SPDX-2.3 checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
