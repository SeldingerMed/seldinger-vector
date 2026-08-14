# OR-Audit

Vendor-neutral robotic surgical skill and safety attestation.

> **Status: v0.1 alpha, pre-Phase-0.** This is engineering scaffolding for the
> product described in [`docs/PLAN.md`](docs/PLAN.md). The plan's Phase 0 gates
> — demand, legal holdability, annotation economics, data rights — are **not
> cleared**, and several load-bearing claims are open verification items
> (`PLAN.md` section V). Nothing here should be read as a validated product or
> as legal, regulatory, or clinical advice.

## What this is

A platform that takes robotic surgical video, evaluates it against published
safety and skill rubrics, and emits an attestation artifact a credentialing
body can actually hold.

The thesis is **not** "we can score robotic surgery" — the robot vendors are
already doing that. It is that an *independent* score is worth paying for
precisely because a manufacturer's score of performance on its own robot is
structurally conflicted, is confined to one platform, and cannot serve as
third-party evidence. See `PLAN.md` sections 1 and 4.

## Architectural commitments

These are enforced in code, not left to convention. They come straight from the
plan and should not be relaxed without amending it.

| Commitment | Where | Why |
|---|---|---|
| Video is required; kinematics never blocks | `domain.Episode` | Kinematics is vendor-gated (V-1). A cross-platform scorer cannot depend on a signal controlled by the parties it competes with. |
| De-identification is a gate, not a flag | `domain.MediaAsset.require_readable` | Only attested media may reach perception, scoring, reporting, or export (§8). |
| The score vector never implicitly collapses | `scoring` (later phase) | Hard safety gates must not average into soft skill scores (§7.1). |
| Abstention is a required output class | `Determination.INDETERMINATE`, `GateStatus.NOT_ASSESSABLE` | A scorer that cannot decline gets forced into false confidence where liability concentrates (§7.2). |
| Every artifact is versioned and chained | `audit.AuditTrail` | The record must be defensible under challenge (§7.3). |
| Attestation requires bytes, not assertion | `deid.redact` | The pipeline hashes what its writer wrote; no caller supplies the digest. A status settable by assertion protects nothing (§8). |
| Platform is data, never a branch | `domain.RobotPlatform` | Vendor-specific behaviour belongs in ingestion adapters only. |
| Gates cannot collapse to a scalar | `scoring.SafetyGateSet` | `float()`, `int()` and `bool()` raise. Hard gates never average into soft scores (§7.1). |
| A gate that cannot see cannot clear | `GateStatus.NOT_ASSESSABLE` | Missing or low-confidence evidence never reads as a pass. |
| ICC form is named, averaging refused | `metrics.icc_2_1` | An unqualified ICC is a family of numbers. Average-measures raises (§13). |
| The agreement target is relative | `metrics.AgreementGate` | The panel is the ceiling; an absolute target either demands superhuman consistency or accepts noise. |
| Binary proficiency is primary | `SkillScore` | GEARS alone is not a result (§13). |

## Layout

```
src/or_audit/
  domain/    entities, closed vocabularies, invariants (no I/O)
  audit/     canonical serialization, tamper-evident append-only trail
  media/     frame access behind a FrameSource protocol
  ingest/    manifests into episodes; kinematics/video alignment
  deid/      detectors, policy, redaction plans, attestation
  perception/ observation vocabulary and backend protocol
  scoring/   hard safety gates; binary proficiency and GEARS
  metrics/   ICC(2,1), Fleiss kappa, the section 13 agreement gate
docs/PLAN.md the product plan this implements
```

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"

uv run pytest              # tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy                # types
```

CI runs lint, format check, mypy, and the test matrix on 3.11–3.13 with a
coverage floor. All must pass before merge.

## Licence

UNLICENSED — proprietary to SeldingerMed. Not for distribution.
