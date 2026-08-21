# SurgEval

[![CI](https://github.com/SeldingerMed/seldinger-vector/actions/workflows/ci.yml/badge.svg)](https://github.com/SeldingerMed/seldinger-vector/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

Auditable evaluation infrastructure for procedural medical AI.

SurgEval turns "the agent scored X" into evidence you can re-derive. The design rests on one rule: the thing being measured does not grade itself. A task pins down what its world requires, an agent declares which capabilities it can actually run, and a task-owned verifier scores each trial from labels the agent never sees. Every trial is written as replayable, content-addressed evidence. The reward you export is recomputed from the authoritative vector, not reported by the agent, and a mismatched head, package digest, or projection is refused outright.

That rule matters in procedural medicine, where a false pass is worse than no signal. Reporting a metric is cheap. Producing a trace another process can re-derive to the same number is not. SurgEval makes the second the requirement.

What you get:

- Versioned contracts instead of ad-hoc prompts. Tasks and agents are package-authored and content-pinned, so a run means the same thing next week and results compare across labs.
- Scoring the agent cannot launder. Agents and task verifiers run in separate processes over a JSON-lines protocol; the agent never sees oracle labels.
- One trace vocabulary across closed-loop, interactive, single-turn, and counterfactual modes, with hard gates kept separate from typed metrics.
- Replay that refuses on mismatch. A vector is reconstructed from its stored trace through the bundled verifier, and any drift in head, package digest, or projection fails the replay.
- A public registry of tasksets and agents, pinned by git ref and content digest, so published runs carry the same trust boundary as local ones.

The v0.3 kernel implements all four harness modes without procedure-specific runner branches. Install the CLI as `surgeval`; `vector` and `or-audit` remain one-release aliases.

## Requirements

The `surgeval` harness itself is CPU-only. It depends on pydantic, numpy, and cloudpickle. Binding, the subprocess agent protocol, verifiers, replay, and export run with no GPU, CUDA, or ROCm dependency. You need Python 3.11 or newer.

A GPU is relevant only to the agents or worlds you evaluate, and those are not part of this package. A vision or RL agent that loads a model needs its own runtime (for example PyTorch with CUDA), sized to the model. The closed-loop Lumen policy path installs `seldinger-lumen` separately; whether that world and its policy run on CPU or want a GPU depends on the specific agent and host, and the harness itself imposes no GPU requirement.

## Core model

| Object | Contract |
|---|---|
| Task | `instruction.md` + pinned world + `InterfaceSpec` + `HarnessSpec` + task-owned verifier |
| Taskset | Versioned collection of tasks with one declared headline metric |
| Agent | `org/name@version` package with one or more `CapabilitySpec` declarations and a pinned runtime descriptor |
| Trial | Typed `ProceduralTrace` + hard gates + typed metrics + optional declarative projection |
| Job | Cartesian product or one bound task-agent pair with portable package copies and a content head |

Interfaces state required interaction mode, protocol version, observation schemas, action/output schemas, and features. Interface IDs and agent kinds are package-authored slugs; capabilities must satisfy every requirement. Binding never switches on procedure names or a closed agent taxonomy.

Four harness modes are implemented:

- `closed-loop`: observation → action → world transition.
- `interactive`: ordered observations → stateful multi-turn outputs → terminal scoring context.
- `single-turn`: task input → structured output, including abstention and uncertainty.
- `counterfactual`: procedural state + candidate interventions → consequence ranking or prediction.

Every mode emits the same typed trace vocabulary: observations, outputs, actions, transitions, safety state, uncertainty, failure, recovery, handoff, tool events, timing, and evidence references.

## Execution boundary

Package Python does not execute in the SurgEval process by default. Local agents and task verifiers use a persistent JSON-lines subprocess protocol with request IDs, timeouts, malformed-output refusal, exit-status capture, and explicit process cleanup. `trusted-in-process` exists only as an explicit runtime kind for controlled test doubles. Runtime descriptors also represent pinned container, Hugging Face, and OpenAI-compatible identities; v0.3 locally executes the subprocess and trusted-test kinds.

The agent receives only task inputs or observations. Labels and other oracle evidence are passed separately to the task-owned verifier.

## Vector semantics

Metrics declare their type and aggregation rule:

- Boolean: true/false counts and assessed rate.
- Continuous: unit, direction, mean, minimum, and maximum.
- Categorical: declared categories and counts.
- Unassessable: `null`, counted separately for every metric type.

Hard gates remain separate. `TrialVector` raises on implicit `float`, `int`, or `bool` conversion.

RL exports use a task-declared `ProjectionSpec`. A projection is data, not Python: source metric, guard metrics, gate-failure behavior, gate-unassessable behavior, output values, version, and rule digest. Export recomputes every value from the authoritative vector and writes the complete rule plus its digest beside each reward.

## Install and run locally

The fastest path is the quickstart script, which creates a venv, installs the
package, and runs the video, counterfactual, and CVS reference examples that
need nothing beyond a CPU:

```bash
./scripts/quickstart.sh
```

To install by hand:

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
```

## Run the packaged reference paths

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"

# Closed-loop Lumen policy
uv run surgeval bind docs/examples/tasks/lumen-nav-safe \
  docs/examples/agents/seldingermed-lumen-linear
uv run surgeval run -t docs/examples/tasks/lumen-nav-safe \
  -a docs/examples/agents/seldingermed-lumen-linear -n 3 \
  --out /tmp/vector-lumen

# Procedural-video structured prediction with abstention
uv run surgeval run -t docs/examples/tasks/video-nextstep \
  -a docs/examples/agents/example-video-predictor \
  --out /tmp/vector-video

# Laparoscopic CVS identification through the video modality adapter
uv run surgeval bind docs/examples/tasks/laparoscopic-cholec-cvs \
  docs/examples/agents/example-cvs-detector
uv run surgeval run -t docs/examples/tasks/laparoscopic-cholec-cvs \
  -a docs/examples/agents/example-cvs-detector \
  --out /tmp/vector-cvs
uv run surgeval replay /tmp/vector-cvs

# Counterfactual world-model consequence ranking
uv run surgeval bind docs/examples/tasks/counterfactual-recovery \
  docs/examples/agents/example-counterfactual-world-model
uv run surgeval run -t docs/examples/tasks/counterfactual-recovery \
  -a docs/examples/agents/example-counterfactual-world-model \
  --out /tmp/vector-counterfactual
uv run surgeval replay /tmp/vector-counterfactual
```

Reward export (`export-rl`) applies the task's declared `ProjectionSpec`. Export
recomputes every value from the authoritative vector and refuses to run against
a job whose world reports no attested simulation backend
(`world_engine.backend="unknown"`), so a counterfactual job must be re-run
against a real backend before its rewards can be exported; see the project
contract in `src/or_audit/eval/export_rl.py`.

Tasksets use the canonical v0.3 verb:

```bash
uv run surgeval tasksets validate docs/examples/tasksets/counterfactual-recovery-v1
uv run surgeval run -s docs/examples/tasksets/counterfactual-recovery-v1 \
  -a docs/examples/agents/example-counterfactual-world-model \
  --out /tmp/vector-taskset
```

`datasets` and `-d/--dataset` remain input aliases for v0.2 automation during migration.

## Public registry

Published tasksets and agents live in [`SeldingerMed/seldinger-tasks`](https://github.com/SeldingerMed/seldinger-tasks). Vector loads `registry.json` from that repo by default; packages are pinned by git ref and content digest.

```bash
# List published packages (HTTPS index only)
uv run surgeval tasksets list
uv run surgeval agents list

# Pull a verified package into a local directory
uv run surgeval agents pull example/video-predictor@0 --out ./packages

# Run against registry references without cloning the harness repo
uv run surgeval run -d seldingermed/video-nextstep@0 \
  -a example/video-predictor@0 \
  --out ./runs/video-nextstep
uv run surgeval replay ./runs/video-nextstep
```

Override the index with `--registry` (local path, `file://` path, or HTTPS URL). Checkouts cache under `~/.cache/surgeval/registry`.

## Artifacts

Each job contains:

- `bundle/task` and `bundle/agent`: exact packages covered by tree digests.
- `bundle.json`: package and runtime identity.
- `config.json`: interface, harness mode, pins, and run count.
- `result.json`: authoritative vectors, typed traces, projection digests, and artifact head.
- `trial-*/trajectory.json`: typed procedural evidence.
- `trial-*/projection.json`: derived projection value, identity, and rule digest.
- `scorecard.json`, `.md`, `.html`: separate gate and typed-metric aggregation plus interface, mode, runtime, projection, package, and artifact identities.

Replay reconstructs each vector from its stored trace through the bundled task verifier before rerunning the world or model. A mismatched vector, package digest, projection, or result head is refused.

## v0.2 package migration

The loader deterministically normalizes existing packages:

| v0.2 | v0.3 |
|---|---|
| task `port` | `InterfaceSpec` + matching `HarnessSpec` |
| agent `port` | `CapabilitySpec` |
| `DatasetSpec` / `dataset.toml` | `TasksetSpec` / `taskset.toml` |
| untyped verifier metric | inferred boolean or continuous `MetricSpec` |
| entrypoint fields | local subprocess `RuntimeDescriptor` |
| tuple-of-dicts trajectory | `ProceduralTrace` with preserved legacy evidence |

The in-tree task and agent examples declare v0.3 contracts directly. External v0.2 packages continue to load and replay through the compatibility adapter.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

Implementation plan and migration details: [`docs/V0.3.md`](docs/V0.3.md). Product architecture and invariants: [`docs/BUILD.md`](docs/BUILD.md). Evaluation rationale: [`docs/ASSESSMENT.md`](docs/ASSESSMENT.md). Dataset licensing and data usage: [`docs/DATASETS.md`](docs/DATASETS.md). Conformance and claim ledger: [`docs/CONFORMANCE.md`](docs/CONFORMANCE.md).

## Publishing to PyPI

The release pipeline lives in `.github/workflows/publish.yml`. It builds the sdist and wheel, runs the identity and license gates, writes an SPDX SBOM, attests the built bytes twice (a GitHub Sigstore attestation plus a PEP 740 attestation stored on PyPI), and verifies PyPI serves the same digests before creating the GitHub Release. It uses PyPI trusted publishing (OIDC), so no password or API token sits in the repository.

To cut a release, bump `version` in `pyproject.toml`, then either push a `vX.Y.Z` tag or dispatch the workflow from `main`:

```bash
git tag v0.3.0 && git push origin v0.3.0
# or
gh workflow run publish.yml -f version=0.3.0
```

To publish from CI, create a pending trusted publisher for the `surgeval` project on PyPI with owner `SeldingerMed`, repository `seldinger-vector`, workflow file `publish.yml`, and environment `release`. Until that mapping exists, install from source with `uv pip install -e ".[dev]"` or point pip at the wheel produced by `uv build`.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
