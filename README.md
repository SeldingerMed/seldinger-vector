# OR-Audit

Vector verifier and attestation kernel for procedural skill, safety, and
technical-AI evaluation.

> **Status: v0.1 alpha.** Engineering scaffolding for a Harbor analog:
> evaluate procedural medical agents in physics and image environments, with
> a vector verifier that cannot hide injury inside a reach metric. Lumen is
> the first world, not the product: lap chole, robotic suturing, endoscopy,
> and endovascular work all sit on the same task contract.
>
> - **Wedge (this is the product):** [`docs/BUILD.md`](docs/BUILD.md) — Harbor
>   analog for *all* image-guided procedural eval. First *runnable* ship is
>   Lumen `safe_success`; fixtures for the other families already load.
> - **Gated mode:** [`docs/PLAN.md`](docs/PLAN.md) — named-surgeon
>   credentialing. Phase 0 is **not** cleared and does **not** block the wedge.
> - **Context:** [`docs/ASSESSMENT.md`](docs/ASSESSMENT.md) — straw-man /
>   steel-man and pre-deployment test layers.
>
> Kernel invariants (no scalar collapse, required abstention, de-id as a gate,
> pinned audit chain) are shared. Nothing here is a validated product or
> legal, regulatory, or clinical advice.

## What this is

A harness that takes a procedural episode (today: synthetic endoscopic video;
intended: sim, public video, de-identified clinical video), runs it through
hard safety gates and soft skill metrics that **cannot be averaged into each
other**, and emits a versioned, contestable artifact.

The product is the Harbor-shaped eval harness in `BUILD.md`: a sandbox for
technical AI on **every image-guided procedure**. Who the subject is (model vs
surgeon) is a mode. Which procedure (wire, scope, lap chole, suture) is a
dataset. `or-audit tasks validate` is the first Harbor verb that actually runs.

The thesis that survives both readings is **not** "we can score robotic
surgery" — vendors already do that — but that an independent *vector*
verifier, able to abstain, is worth paying for when self-issued scores are
structurally conflicted. Who the subject is (surgeon vs policy) is a mode,
not a rewrite.

## Architectural commitments

These are enforced in code, not left to convention. They come straight from the
plan and should not be relaxed without amending it.

| Commitment | Where | Why |
|---|---|---|
| Video is required; kinematics never blocks | `domain.Episode` | Kinematics is vendor-gated (V-1). A cross-platform scorer cannot depend on a signal controlled by the parties it competes with. |
| De-identification is a gate, not a flag | `domain.MediaAsset.require_readable` | Only attested media may reach perception, scoring, reporting, or export (§8). |
| The score vector never implicitly collapses | `scoring.ScoreVector` | Hard safety gates must not average into soft skill scores (§7.1). |
| Abstention is a required output class | `Determination.INDETERMINATE`, `GateStatus.NOT_ASSESSABLE` | A scorer that cannot decline gets forced into false confidence where liability concentrates (§7.2). |
| Every artifact is versioned and chained | `audit.AuditTrail` | The record must be defensible under challenge (§7.3). |
| Attestation requires bytes, not assertion | `deid.redact` | The pipeline hashes what its writer wrote; no caller supplies the digest. A status settable by assertion protects nothing (§8). |
| Analysis and attestation are different claims | `DeidPolicy.guarantees_overlay_coverage` | Attesting needs a recall bound under the ceiling **and** a recorded measurement backing it. The default cannot attest while V-10 is open; a coarse grid cannot attest even with one. |
| Platform is data, never a branch | `domain.RobotPlatform` | Vendor-specific behaviour belongs in ingestion adapters only. |
| Gates cannot collapse to a scalar | `scoring.SafetyGateSet` | `float()`, `int()` and `bool()` raise. Hard gates never average into soft scores (§7.1). |
| A gate that cannot see cannot clear | `GateStatus.NOT_ASSESSABLE` | Missing or low-confidence evidence never reads as a pass. |
| ICC form is named, averaging refused | `metrics.icc_2_1` | An unqualified ICC is a family of numbers. Average-measures raises (§13). |
| The agreement target is relative | `metrics.AgreementGate` | The panel is the ceiling; an absolute target either demands superhuman consistency or accepts noise. |
| Binary proficiency is primary | `SkillScore` | GEARS alone is not a result (§13). |
| The collapse is owned, not avoided | `decision.DecisionRule` | Credentialing ends in a binary act; if we don't collapse the vector every hospital invents its own (§7.2). |
| Contestability ships with v1 | `decision.DecisionRecord` | Right of access, appeal, surfaced rater disagreement, right of response (§7.3). |

## Layout

```
src/or_audit/
  eval/      Harbor-shaped tasks, datasets, trial vectors (BUILD.md P0)
  domain/    entities, closed vocabularies, invariants (no I/O)
  audit/     canonical serialization, tamper-evident append-only trail
  media/     frame access behind a FrameSource protocol
  ingest/    manifests into episodes; kinematics/video alignment
  deid/      detectors, policy, redaction plans, attestation
  perception/ observation vocabulary and backend protocol
  scoring/   hard safety gates; binary proficiency and GEARS
  metrics/   ICC(2,1), Fleiss kappa, the section 13 agreement gate
  decision/  pre-registered decision rule, contestation, disclosure
docs/BUILD.md       Harbor-for-medicine build plan (the wedge)
docs/PLAN.md        credentialing-mode spec (gated; does not block B)
docs/ASSESSMENT.md  straw-man / steel-man and pre-deployment test layers
docs/examples/      P0 fixtures across families: endovascular, endoscopy,
                    laparoscopy, robotic suturing
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

uv run or-audit tasks validate docs/examples/tasks/lumen-nav-safe
uv run or-audit datasets validate docs/examples/datasets/lumen-nav-v0
```

CI runs lint, format check, mypy, and the test matrix on 3.11–3.13 with a
coverage floor. All must pass before merge.

## Licence

UNLICENSED — proprietary to SeldingerMed. Not for distribution.
