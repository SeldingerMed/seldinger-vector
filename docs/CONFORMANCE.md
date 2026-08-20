# Conformance and Claim Ledger

This document records what SurgEval verifies, what it does not verify, and what claims a conformant result may make. It is the authoritative reference for interpreting scorecard outputs.

## Conformance levels

### Task conformance

A task package conforms when:

1. `task.toml` validates against the schema: required fields present, enums in the closed vocabulary, headline metric exists in the metric set.
2. `instruction.md` exists and states the task's purpose, data, and claim boundary.
3. The verifier (`verifier.py`) loads and returns a `VerifierRuntime` with declared metrics and gates.
4. Input and label files exist and are non-empty (for non-simulation tasks).
5. `WorldSpec` is valid: `world_pin` is non-empty, `world_kind` is a known enum, `synthetic_stub` is declared.
6. The interface spec declares `id`, `interaction_mode`, `protocol_versions`, and (when applicable) `modalities`.

**What task conformance does not verify**: the clinical validity of the task design, the appropriateness of the gates for a real clinical workflow, or the correctness of the oracle labels. These are the task author's responsibility.

### Agent conformance

An agent package conforms when:

1. `agent.toml` validates against the schema: `id`, `agent_version`, `kind`, `weights_pin`, `weights_path`, at least one `CapabilitySpec`, and a `RuntimeDescriptor`.
2. The weights file at `weights_path` exists and its SHA-256 digest matches `weights_pin`.
3. The runtime entrypoint loads and returns the expected runtime type (`PolicyRuntime` or `PredictorRuntime`).
4. Each `CapabilitySpec` declares `interface`, `interaction_modes`, `protocol_versions`, and `schema_wildcard` (defaulting to `false`).

**What agent conformance does not verify**: the model's clinical performance, its generalization to populations not represented in the dataset, or its safety in a real clinical deployment. These require evaluation results, not package validation.

### Binding conformance

A binding (task-agent pair) conforms when:

1. The agent declares at least one `CapabilitySpec` whose `interface` matches the task's `InterfaceSpec.id`.
2. The capability's `interaction_modes` include the task's `HarnessSpec.interaction_mode`.
3. The capability's `protocol_versions` intersect with the interface's `protocol_versions`.
4. If `schema_wildcard = false`, the capability's observation and output schemas must satisfy the interface's declared schemas. If `schema_wildcard = true`, the binding is accepted but `binding_mode: "wildcard"` is stamped in the config and scorecard.

**What binding conformance does not verify**: semantic compatibility between the agent's outputs and the task's expectations beyond the declared schema. A wildcard binding is structurally valid but does not prove the agent produces meaningful outputs for this task.

### Result conformance

A job result conforms when:

1. Every trial has a `TrialVector` with gates and metrics matching the task's verifier declaration.
2. Every gate has a status of `pass`, `fail`, or `not-assessable` (never `null` for gates that the verifier assesses).
3. The `head` (SHA-256 of the canonical job payload) is present and matches recomputation.
4. The config records `task_dir` and `agent_dir` as relative paths within the bundle.
5. The bundle's task and agent tree digests match the config's `task_digest` and `agent_digest`.
6. If the task uses a simulation backend, the `world_engine` provenance is recorded in the config and scorecard.
7. If `synthetic_stub = true`, the scorecard displays a synthetic banner and the result is not exportable as RL training data.

**What result conformance does not verify**: that the agent was the best possible model, that the gates caught every failure mode, or that the result generalizes beyond the evaluated data.

## Claim ledger

| Claim | Verified by | Limitation |
|---|---|---|
| "Agent X binds to task Y" | `surgeval bind` | Structural compatibility only; not semantic correctness. |
| "Agent X produced result Z on task Y" | Job config + result head + bundle digests | Proves the agent ran; does not prove the result is clinically meaningful. |
| "Result Z is reproducible" | `surgeval replay` | Proves the vector reconstructs from the trace; depends on the bundled task and agent packages being available. |
| "Gate G passed" | Verifier output in the trial vector | Proves the gate's condition was met for this trial; does not prove the gate is sufficient for clinical safety. |
| "Headline metric H = v" | Verifier-computed metric in the vector | Proves the metric value for this evaluation; does not prove the metric is the right measure of performance. |
| "Result Z is exportable as RL data" | `surgeval export-rl` with projection | Only for non-synthetic-stub runs; the projection rule is task-declared and versioned. |
| "World engine E was used" | `world_engine` provenance in config | Records which engine and whether it was a synthetic stub; does not validate the engine's physics fidelity. |
| "Agent identity is A@v+pin" | `agent.toml` weights_pin + bundle digest | Proves the agent package is pinned; for SDK-synthesized agents, the pin is derived from the model's serialized bytes. |

## What SurgEval does not claim

1. **Clinical validation**: SurgEval is evaluation infrastructure, not a clinical validation framework. A passing result does not certify a model for clinical use.
2. **Gate sufficiency**: The platform enforces that gates are evaluated and reported. It does not certify that the declared gates are sufficient for any clinical scenario.
3. **Dataset representativeness**: The platform pins datasets by digest. It does not assess whether a dataset is representative of any patient population.
4. **Model robustness**: A result on one task does not imply robustness on related tasks, out-of-distribution inputs, or adversarial conditions.
5. **Simulation fidelity**: The platform records the simulation engine and stamps synthetic runs. It does not validate that the simulation accurately models any real procedure.

## Interpretation guide

A conformant SurgEval result supports the following chain of reasoning:

1. **What ran**: The config and bundle identify the exact task, agent, and runtime.
2. **What happened**: The trial vectors record gates, metrics, and typed traces.
3. **What it means**: The claim boundary in `instruction.md` scopes what the result asserts.
4. **Whether it holds**: Replay reconstructs the vectors from traces through the bundled verifier, and the result head catches any mismatch.

Everything beyond this chain — clinical interpretation, regulatory submission, deployment decisions — is the consumer's responsibility, not a SurgEval claim.
