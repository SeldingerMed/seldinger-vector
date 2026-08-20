# SurgEval OSS Governance

This document is the operating agreement for how SurgEval stays neutral,
commercially redistributable, and verifiably authored — the "P1 packaging"
surface. It is read by contributors and maintainers; the enforceable parts are
backed by CI gates and scripts listed below.

SurgEval's product thesis is in [`docs/BUILD.md`](docs/BUILD.md). Section 1.4
("What we will not build") is normative: e.g. *a heuristic catalog of
procedures* — if we have to name the anatomy to load the task, the sandbox has
already failed. Everything in this document serves that line.

## 1. Identity consolidation

Two importable top-level packages ship in one wheel:

- `or_audit` — the internal implementation (layered `domain`/`audit` modules).
- `surgeval` — the public SDK shim re-exporting the stable surface.

The public identity is **`surgeval`**. They must never diverge in version.
`scripts/check_identity.py` enforces a single version across `pyproject`,
`or_audit.version.PACKAGE_VERSION`, and `surgeval.__version__`, and asserts the
built wheel contains both trees. On release tags it also gates
`--expected-version` so a tag never publishes a wheel whose metadata says a
different version. All attestation artifacts record this version (PLAN §7.3).

## 2. Third-party dependency policy (3P policy)

Adding any third-party package is a review event, not a mechanical step:

- **Runtime deps must be commercially friendly.** `scripts/check_license_allowlist.py`
  computes the runtime dependency closure from `pyproject` root deps through
  `uv.lock` and rejects any package without an explicit reviewed
  `THIRD_PARTY` license entry, or any license outside the permissive allowlist
  (`MIT`, `BSD-2/3`, `Apache-2.0`, `PSF-2.0`, `0BSD`, `Zlib`, `ISC`, `CC0-1.0`).
  A new runtime dependency failing here is the intended gate.
- **No copyleft in runtime.** GPL/AGPL are disallowed outright; MPL-2.0
  (file-level copyleft) is tolerated only for dev/build tooling, never runtime.
- **Pin, don't float.** Dev and runtime pins are reviewed. Registry-index
  packages are additionally pinned by git ref + content digest (see §5).
- **Commercial allowlist.** The runtime allowlist doubles as the commercial
  allowlist: any dependency deep-linked into a for-profit artifact must be on
  it. If a proposed dep is not, it is rejected or moved behind an optional
  extra gated separately.

### Record of current runtime dependencies

| name | version | license |
|---|---|---|
| pydantic | 2.13.4 | MIT |
| pydantic-core | 2.46.4 | MIT |
| annotated-types | 0.8.0 | MIT |
| typing-inspection | 0.4.4 | MIT |
| typing-extensions | 4.16.0 | PSF-2.0 |
| numpy | 2.4.6 / 2.5.2 | BSD-3-Clause AND MIT AND 0BSD AND Zlib AND CC0-1.0 |
| cloudpickle | 3.1.2 | BSD-3-Clause |

(NVIDIA-hosted `warp-lang` is an optional/`lumen`-extra integration, not a
runtime core dependency; it is not part of the shipping runtime closure.)

## 3. IP and licensing

- Repository license: **Apache-2.0** (see `LICENSE`).
- `NOTICE` attributes the product: *Seldinger Vector (OR-Audit), Copyright 2026
  SeldingerMed*. Keep it accurate; it is the legal attribution surface.
- Distribution metadata: `pyproject [project]` declares `name = "surgeval"` and
  the license file; do not add a second license grant.
- Contributors agree to license their contributions under Apache-2.0 by
  submitting a PR (see `CONTRIBUTING.md`).

## 4. SBOM

`scripts/gen_sbom.py` resolves `uv.lock` into an **SPDX 2.3** JSON document with
unique per-variant package IDs, dependency edges, license conclusions from the
reviewed table `scripts/check_license_allowlist.THIRD_PARTY`, and a
`documentNamespace` keyed by the actual lock bytes.

`scripts/validate_sbom.py` validates it against the official SPDX 2.3 JSON
schema (CI downloads it) and falls back to a structural check.

CI generates + validates the SBOM on every release (`.github/workflows/publish.yml`)
and on demand (`.github/workflows/sbom.yml`), and attaches it to the GitHub
Release.

## 5. Publisher signatures (release-artifact layer)

SurgEval does **not** ship a dormant in-band signing key — unkeyed signature
fields are not verification. Publisher provenance is established and verified
out-of-band at the asset layer:

1. **GitHub Sigstore build attestation.** `actions/attest-build-provenance`
   records SLSA provenance for the built wheel+sdist; `gh attestation verify`
   cryptographically confirms the artifact was built by this repository's
   workflow against Sigstore's public-good root.
2. **PyPI trusted publishing (OIDC)** — secretless; the workflow identity is
   proven to PyPI by GitHub's OIDC id-token.
3. **PyPI PEP 740 attestation** — `pypa/gh-action-pypi-publish` records a
   per-file Sigstore attestation on PyPI; `scripts/check_pypi_attestation.py`
   confirms PyPI serves the exact built digest with the bundle present (the
   crypto verification of the same bytes is the GitHub attestation above).

The release job is fail-closed: publish runs only after attestation succeeds,
and the GitHub Release is created only after the PyPI check passes.

**Task-registry package signing is a separate trust domain.** Pinning
taskset/agent packages from `SeldingerMed/seldinger-tasks` is currently by git
ref + tree content digest at materialize time (see `src/or_audit/eval/registry.py`).
Dormant "signature" fields will not be added as a downgrade path; if
cryptographic signing of the *registry index itself* is introduced, the trust
anchor (public key / key-id) must be pinned out-of-band in the installed
package or user config — never shipped inside the remotely fetched index — and
load must fail closed on unsigned or invalid indexes.

## 6. Neutral tasksets

SurgEval is the neutral evaluation layer; first-party Seldinger identity must
not be load-bearing.

- Bundled example tasksets/agents use **neutral** identities (`example/…`), not
  `seldingermed/…`. See `docs/examples/tasksets/consequence-rank-v1` (id
  `example/consequence-rank`) as the seed.
- First-party Seldinger tasksets belong in the `SeldingerMed/seldinger-tasks`
  registry, not as bundled seeds.
- Every taskset must declare `phi_class` and a headline, and must not require a
  procedure-name taxonomy to load (BUILD §1.4).
- PHI-class handling is declarative in the taskset/agent contracts; never
  hard-code PHI-identifying content.

## 7. Enforcement / operational notes

- Contributor gates: `CONTRIBUTING.md`; security processes: `SECURITY.md`;
  community standards: `CODE_OF_CONDUCT.md`.
- The `release` **environment** and PyPI project must be configured for trusted
  publishing (OIDC) by maintainers before the first tagged release; no
  long-lived upload secrets exist.

## 8. Known gaps (tracked, not declared done)

- **Registry index signing** requires a real out-of-band SeldingerMed key that
  does not exist in this repository's scope. Until it ships, registry trust is
  content-digest + git-ref pinning (§5). This is an explicit, named gap, not a
  pretended feature.
