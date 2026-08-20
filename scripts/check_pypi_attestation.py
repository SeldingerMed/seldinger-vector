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

Run: ``uv run python scripts/check_pypi_attestation.py <version>
<RELEASE_DIGESTS.txt> [--publisher-repo owner/repo]``

Every published wheel and sdist must appear in ``RELEASE_DIGESTS.txt`` (the
``sha256sum`` output recorded in the release job) with a matching digest and a
PEP 740 attestation bundle; anything extra, missing, or mismatched fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ACCEPT = "application/vnd.pypi.integrity.v1+json"


def _load_expected(digest_file: Path) -> dict[str, str]:
    """Parse ``sha256sum`` output (``<sha>  <path>``) into {basename: sha256}."""
    expected: dict[str, str] = {}
    for line in digest_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        sha, _, path = line.partition("  ")
        expected[Path(path).name] = sha.strip()
    return expected


def _open(url: str, accept: str | None = None) -> dict:
    req = urllib.request.Request(url)
    if accept:
        req.add_header("Accept", accept)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except Exception as exc:
        raise SystemExit(f"failed to fetch {url}: {exc}") from exc


def check(version: str, expected: dict[str, str], publisher_repo: str | None) -> int:
    index = _open(f"https://pypi.org/pypi/surgeval/{version}/json")
    urls = index.get("urls", [])
    if not urls:
        print(f"FAIL: no files for surgeval=={version} on PyPI", file=sys.stderr)
        return 1

    dist_files = [u for u in urls if u["filename"].endswith((".whl", ".tar.gz"))]
    if not dist_files:
        print(f"FAIL: no wheel/sdist for surgeval=={version} on PyPI", file=sys.stderr)
        return 1

    failures: list[str] = []
    checked = 0
    for u in dist_files:
        fn = u["filename"]
        exp = expected.get(fn)
        if exp is None:
            failures.append(f"PyPI serves {fn} but it is not in the built set")
            continue
        served = (u.get("digests") or {}).get("sha256", "")
        if served != exp:
            failures.append(f"PyPI {fn} digest {served} != built digest {exp}")
            continue
        qfn = urllib.parse.quote(fn)
        integrity = _open(
            f"https://pypi.org/integrity/surgeval/{version}/{qfn}/provenance",
            accept=ACCEPT,
        )
        bundles = integrity.get("attestation_bundles") or []
        if not bundles:
            failures.append(f"PyPI {fn} has no PEP 740 provenance bundles")
        elif publisher_repo and not any(
            (b.get("publisher") or {}).get("repository") == publisher_repo for b in bundles
        ):
            failures.append(f"no PEP 740 bundle for {fn} published by {publisher_repo!r}")
        else:
            checked += 1

    # Every built distribution file must actually be served — a wheel or sdist
    # missing from the index (e.g. an unattested sdist dropped at upload) must
    # fail even if every served file individually checks out.
    expected_dist = {name for name in expected if name.endswith((".whl", ".tar.gz"))}
    served_names = {u["filename"] for u in dist_files}
    for missing in sorted(expected_dist - served_names):
        failures.append(f"expected built artifact {missing} is not served by PyPI")
    failures.extend(
        f"PyPI serves {name} with no matching built digest"
        for name in sorted(served_names - expected_dist)
    )

    if failures:
        for f in failures:
            print("FAIL:", f, file=sys.stderr)
        return 1

    print(
        f"OK: PyPI serves surgeval=={version} with {checked} wheel/sdist matching built "
        f"digests and PEP 740 bundle(s) present"
        + (f" published by {publisher_repo!r}" if publisher_repo else "")
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("digest_file", help="path to RELEASE_DIGESTS.txt (sha256sum output)")
    ap.add_argument("--publisher-repo", default=None)
    args = ap.parse_args(argv)
    expected = _load_expected(Path(args.digest_file))
    if not expected:
        print("FAIL: no digests parsed from RELEASE_DIGESTS.txt", file=sys.stderr)
        return 1
    return check(args.version, expected, args.publisher_repo)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
