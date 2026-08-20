# Contributing to SurgEval

SurgEval (distribution name `surgeval`, internal package `or_audit`) is the
independent evaluation, benchmarking, and safety-verification layer for
procedural medical AI. It is built such that third parties — including Seldinger
competitors — can contribute tasksets, agents, and adapters without endorsing a
Seldinger first-party taxonomy. Read [`docs/BUILD.md`](docs/BUILD.md) for the
product thesis; section 1.4 ("What we will not build") is normative for what
this repository will never become.

## Ground rules

- **Neutral, not Seldinger-branded.** Bundled example tasksets/agents live under
  neutral identities (`example/…`), never `seldingermed/…`. First-party
  Seldinger packages belong in the `SeldingerMed/seldinger-tasks` registry, not
  as load-bearing knowledge in this repo.
- **No procedure-name taxonomy as a load-bearing type.** Never dispatch closed
  procedure enums (`ModalityKind`, etc.) as routing keys. Routing and data-shape
  identity go through **schema/plugin IDs** (`StreamSpec.schema_id`,
  `StreamSpec.adapter`, pinned content digests). A task must never require
  naming the anatomy to load.
- **Evidence-backed gates.** Every verifier gate must be resolvable from
  evidence and kernel-hashed in its reasons. Abstention (missing evidence)
  is `NOT_ASSESSABLE`, never a hard error. Reuse `or_audit.audit.canonical.digest`
  (RFC-8785); reasons carry binding IDs and digests, never raw signal values.
- **Streams are authoritative.** There is no dual-routing fallback to legacy
  `modalities`. See the modality/stream contract in
  [`src/or_audit/eval/contracts.py`](src/or_audit/eval/contracts.py).

## Third-party contribution policy

Consult [`docs/OSS_GOVERNANCE.md`](docs/OSS_GOVERNANCE.md) before adding any
dependency. In short:

- Every runtime dependency must carry a commercial-friendly license on the
  reviewed allowlist (`scripts/check_license_allowlist.py` fails otherwise).
- No copyleft (GPL/AGPL) runtime dependencies; MPL-2.0 is dev/build only.
- Registry-index packages are pinned by git ref **and** content digest; do not
  weaken those checks.

## Development setup

Requires Python ≥ 3.11 and `uv`.

```bash
uv sync --all-extras
uv run pytest -q          # full suite, must stay green
uv run ruff check .       # lint
uv run ruff format --check .
uv run mypy               # strict type-check
```

CI runs lint, type-check, and a test matrix over Python 3.11/3.12/3.13.

## Adding a task / taskset / agent

Add it under `docs/examples/…` with a **neutral** identity, and validate it with
the canonical verb:

```bash
uv run surgeval tasksets validate docs/examples/tasksets/<name>-v1
```

Every bundled taskset must declare a `phi_class` and a headline, reference
immutable tasks, and avoid first-party branding. PHI-class rules live in the
taskset/agent contracts — never hard-code PHI-identifying content.

## Releasing (maintainers)

See [`.github/workflows/publish.yml`](.github/workflows/publish.yml). Push a
`v*` tag (or dispatch with a matching version input). The workflow enforces, in
order: version/identity gate, license allowlist, full tests, wheel+sdist build,
SPDX SBOM validation, GitHub Sigstore build attestation + `gh attestation
verify`, then PyPI trusted publishing (OIDC) with PEP 740 attestations, then a
final PyPI digest/attestation check — and only then a GitHub Release. Do **not**
bump versions piecemeal; keep `pyproject`, `src/or_audit/version.py`, and the tag
in lockstep (`scripts/check_identity.py --expected-version` enforces this).

## Standards

- Ruff (line-length 100) and mypy strict via the pydantic plugin are mandatory.
- Follow the existing layering: `domain` (entities/vocabularies, no I/O) and
  `audit` (deterministic serialization + tamper-evident trail) at the bottom.
- Tests must defend observable contracts (behavior, invariants, boundaries,
  failures), not source text.
