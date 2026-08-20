"""Confirm what PyPI serves for a release matches the built artifact + has PEP 740 provenance.

NOTE — scope: this is a PRESENCE and content-consistency check, NOT a standalone
cryptographic signature verification. It confirms the artifact PyPI now serves
(1) is bit-identical to what was built (``sha256`` from ``RELEASE_DIGESTS.txt``)
and (2) carries PEP 740 provenance via the PyPI Integrity API. The
cryptographic Sigstore verification of that provenance is performed by
``gh attestation verify`` over the GitHub build attestations produced by
``actions/attest-build-provenance`` for the same subject bytes (see
``.github/workflows/publish.yml``).

Uses the PyPI Integrity API:
``GET /integrity/<project>/<version>/<filename>/provenance`` with
``Accept: application/vnd.pypi.integrity.v1+json``.

Run: ``uv run python scripts/check_pypi_attestation.py <version> <expected-sha256>
[--publisher-repo owner/repo]``
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request

ACCEPT = "application/vnd.pypi.integrity.v1+json"


def _open(url: str, accept: str | None = None) -> dict:
    req = urllib.request.Request(url)
    if accept:
        req.add_header("Accept", accept)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except Exception as exc:
        raise SystemExit(f"failed to fetch {url}: {exc}") from exc


def check(version: str, expected_sha256: str, publisher_repo: str | None) -> int:
    index = _open(f"https://pypi.org/pypi/surgeval/{version}/json")
    wheel = next(
        (u for u in index.get("urls", []) if u.get("filename", "").endswith(".whl")),
        None,
    )
    if wheel is None:
        print(f"FAIL: no wheel for surgeval=={version} on PyPI", file=sys.stderr)
        return 1

    served_digest = (wheel.get("digests") or {}).get("sha256", "")
    if served_digest != expected_sha256:
        print(
            f"FAIL: PyPI wheel digest {served_digest} != built digest {expected_sha256}",
            file=sys.stderr,
        )
        return 1

    filename = urllib.parse.quote(wheel["filename"])
    integrity = _open(
        f"https://pypi.org/integrity/surgeval/{version}/{filename}/provenance",
        accept=ACCEPT,
    )
    bundles = integrity.get("attestation_bundles") or []
    if not bundles:
        print("FAIL: PyPI Integrity API returned no PEP 740 provenance bundles", file=sys.stderr)
        return 1

    if publisher_repo:
        matched = any(
            (b.get("publisher") or {}).get("repository") == publisher_repo for b in bundles
        )
        if not matched:
            print(
                f"FAIL: no PEP 740 bundle published by {publisher_repo!r} for the served artifact",
                file=sys.stderr,
            )
            return 1

    print(
        f"OK: PyPI serves surgeval=={version} with digest {expected_sha256} and "
        f"{len(bundles)} PEP 740 bundle(s) present"
        + (f" published by {publisher_repo!r}" if publisher_repo else "")
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("expected_sha256")
    ap.add_argument("--publisher-repo", default=None)
    args = ap.parse_args(argv)
    return check(args.version, args.expected_sha256.lower(), args.publisher_repo)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
