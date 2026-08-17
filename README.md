# OR-Audit

Independent evaluation infrastructure for procedural medical AI.

OR-Audit binds versioned task interfaces to declared agent capabilities, executes agents and task-owned verifiers in separate processes, and writes replayable vector evidence. The v0.3 kernel supports closed-loop policies, multi-turn interactive agents, structured single-turn models, and counterfactual world models without adding procedure-specific runner branches.

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

Package Python does not execute in the OR-Audit process by default. Local agents and task verifiers use a persistent JSON-lines subprocess protocol with request IDs, timeouts, malformed-output refusal, exit-status capture, and explicit process cleanup. `trusted-in-process` exists only as an explicit runtime kind for controlled test doubles. Runtime descriptors also represent pinned container, Hugging Face, and OpenAI-compatible identities; v0.3 locally executes the subprocess and trusted-test kinds.

The agent receives only task inputs or observations. Labels and other oracle evidence are passed separately to the task-owned verifier.

## Vector semantics

Metrics declare their type and aggregation rule:

- Boolean: true/false counts and assessed rate.
- Continuous: unit, direction, mean, minimum, and maximum.
- Categorical: declared categories and counts.
- Unassessable: `null`, counted separately for every metric type.

Hard gates remain separate. `TrialVector` raises on implicit `float`, `int`, or `bool` conversion.

RL exports use a task-declared `ProjectionSpec`. A projection is data, not Python: source metric, guard metrics, gate-failure behavior, gate-unassessable behavior, output values, version, and rule digest. Export recomputes every value from the authoritative vector and writes the complete rule plus its digest beside each reward.

## Run the packaged reference paths

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"

# Closed-loop Lumen policy
uv run or-audit bind docs/examples/tasks/lumen-nav-safe \
  docs/examples/agents/seldingermed-lumen-linear
uv run or-audit run -t docs/examples/tasks/lumen-nav-safe \
  -a docs/examples/agents/seldingermed-lumen-linear -n 3 \
  --out /tmp/or-audit-lumen

# Procedural-video structured prediction with abstention
uv run or-audit run -t docs/examples/tasks/video-nextstep \
  -a docs/examples/agents/example-video-predictor \
  --out /tmp/or-audit-video

# Counterfactual world-model consequence ranking
uv run or-audit bind docs/examples/tasks/counterfactual-recovery \
  docs/examples/agents/example-counterfactual-world-model
uv run or-audit run -t docs/examples/tasks/counterfactual-recovery \
  -a docs/examples/agents/example-counterfactual-world-model \
  --out /tmp/or-audit-counterfactual
uv run or-audit replay /tmp/or-audit-counterfactual
uv run or-audit export-rl /tmp/or-audit-counterfactual \
  --projection gated-recovery-v1 \
  --out /tmp/or-audit-counterfactual/rollouts.jsonl
```

Tasksets use the canonical v0.3 verb:

```bash
uv run or-audit tasksets validate docs/examples/tasksets/counterfactual-recovery-v1
uv run or-audit run -s docs/examples/tasksets/counterfactual-recovery-v1 \
  -a docs/examples/agents/example-counterfactual-world-model \
  --out /tmp/or-audit-taskset
```

`datasets` and `-d/--dataset` remain input aliases for v0.2 automation during migration.

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

Implementation plan and migration details: [`docs/V0.3.md`](docs/V0.3.md). Product architecture and invariants: [`docs/BUILD.md`](docs/BUILD.md). Evaluation rationale: [`docs/ASSESSMENT.md`](docs/ASSESSMENT.md).

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
