# BUILD: Harbor for procedural medical AI

**Status:** definitive build plan for Future B (decided).
**Analog:** [Harbor](https://www.harborframework.com/) — evaluate agents in sandboxed environments — specialized for medical world models and technical procedural AI.
**Not:** `PLAN.md` credentialing (Future C). That remains a gated mode on these rails. Do not let hospital ACV block this plan.
**Companions:** [Lumen](https://github.com/SeldingerMed/seldinger-lumen), [AngioStress](https://github.com/SeldingerMed/angiostress-benchmark), [`ASSESSMENT.md`](ASSESSMENT.md).

Harbor’s homepage is one sentence: *evaluate agents in sandboxed environments.* Ours is the same sentence with the sandbox replaced:

> Evaluate medical procedural agents — policies, frozen perception models, VLMs, world models — in physics and image environments, and score them with a vector verifier that cannot hide injury inside a reach metric.

This document is the product spec and the build order. `or_audit.eval` is the kernel that makes the spec load-bearing.

---

## 1. Product

### 1.1 What we sell

Niche infrastructure, not a model and not a hospital app.

1. **A task format** for procedural evals (instruction + world + vector verifier).
2. **A runner** that turns `(task, agent, seed)` into a replayable trial.
3. **Datasets** that are versioned collections of tasks (the analog of Terminal-Bench / SWE-Bench).
4. **An RL interface** that exports trajectories and a *versioned scalar projection* of the vector — never the other way around.
5. **A registry / leaderboard** whose headline is always the safety-aware metric.

Buyers: robot-policy teams, endovascular/endoscopic autonomy groups, medical-CV labs, later assurance programs. They already need third-party numbers they cannot credibly self-issue.

### 1.2 Harbor map (normative)

Harbor’s objects are the right objects. Harbor’s *world* and *reward* are the wrong primitives for medicine. This table is the product. Deviations need a written amendment.

| Harbor | This stack | Keep / change |
|---|---|---|
| `harbor` CLI | `or-audit` CLI | Keep the verbs: `run`, `datasets`, `view` (later) |
| **Task** = instruction + Dockerfile + `tests/` → `reward.txt` | **Task** = `instruction.md` + world spec + vector verifier | Change the world and the verifier |
| **Dataset** = collection of tasks, optional custom metrics | **Dataset** = collection of tasks; custom metrics must not erase the safety vector | Keep |
| **Agent** = Claude Code, OpenHands, Terminus, custom `BaseAgent` | **Agent** = policy checkpoint, frozen model, VLM, (later) panel | Keep the interface, change the population |
| **Environment** = Docker / Daytona / Modal / E2B | **World** = Lumen gym, FrameSource, AngioStress contract. Containers wrap the *runner*, not the patient | Change |
| **Trial** = one agent attempt; “a rollout that produces a reward” | **Trial** = one agent attempt; a rollout that produces a **vector**. Reward is an optional projection | Change |
| **Job** = cartesian product of agents × tasks × attempts | **Job** = same | Keep |
| Registry of published datasets | Registry of published *procedural* datasets | Keep |
| Cloud sandboxes for 1000s of containers | GPU workers for batched Lumen envs; isolated jobs for weights | Change the substrate |
| RL via SkyRL; `reward` + token ids | RL via Gymnasium/SB3/SkyRL; **projection** + trajectory | Change the reward |
| ATIF trajectories (tokens, tools, images) | Procedural trajectories (actions, obs, images, contact, vector) | Keep the idea |
| `harbor view` job browser | Later; JSON artifacts first | Defer UI |

### 1.3 Invariants that Harbor does not have (non-negotiable)

These are already encoded in the credentialing kernel and must hold on every eval path:

1. **Vector, not scalar.** Hard gates and task metrics report separately. `TrialVector` raises on `float`/`int`/`bool`.
2. **Headline is safety-aware.** A dataset that reports only raw reach is invalid. Lumen’s lesson: 100% `success`, 6.7% `safe_success`.
3. **Abstention is a legal outcome** where the oracle cannot see. Physics tasks may set `abstain_ok = false` because contact is decidable. Video/VLM tasks may not.
4. **PHI class is a field, not a hope.** `procedural` / `public` / `deidentified_clinical` / `prohibited`. Clinical video cannot ride a shared eval cluster without a BAA path.
5. **Subject kind is a field.** `policy` / `model` / `human`. Human determinations stay refused until `PLAN.md` Phase 0. This plan does not wait on that.
6. **Oracle kind is a field.** `physics` (Lumen wall), `contract` (AngioStress), `panel` (raters), `script` (Harbor’s `solve.sh`, rare).
7. **Analysis ≠ attestation.** Sim tasks attest nothing. Clinical tasks cannot attest without V-10.
8. **Replay identity** is `(task_id, task_version, agent_identity, seed, world_pin)`. A leaderboard row without this is a blog post.

### 1.4 What we will not build

- Docker as the procedural world (Harbor’s sandbox is our *job isolation*, not our lumen).
- `reward.txt` as the primary interface.
- A general coding-agent harness. We do not compete with Harbor on Terminal-Bench.
- Intraoperative decision support, live gating, robot certification.
- Named-surgeon credentialing UX (Future C).
- Selling labelled clinical video as the business.
- A world-model foundation model of our own as a prerequisite. We evaluate other people’s.

---

## 2. What we already have (so we do not rebuild it)

| Asset | Repo | Maps to Harbor | Gap |
|---|---|---|---|
| Vector verifier, abstention, audit chain, de-id gate, agreement harness | `or-audit` | Verifier + artifact | No task/job/trial objects, no runner, no agent protocol for policies |
| Gym envs `Lumen/Nav*-v0`, `safe_success`, batched Newton, capture/replay | `seldinger-lumen` | Environment + first tasks | Not wrapped as OR-Audit tasks; no pin from this repo |
| Frozen-model DSA/segmentation contracts, release audit | `angiostress-benchmark` | Dataset + contract oracle | Not a task directory |
| Equivariance prior for gauges | `gaugeflow` | Optional agent/model family later | Out of P0–P2 |
| Synthetic video pipeline + credentialing demo | `or-audit` `demo` | A *video* task backend | Still emits a privileging report as if that were the product |

Build *on* these. The missing product is the Harbor glue plus the first published datasets.

---

## 3. Architecture

```
                    ┌──────────── datasets (versioned) ────────────┐
                    │  lumen-nav-v0   angiostress-v0   video-v0    │
                    └──────────────────────┬───────────────────────┘
                                           │ tasks
┌─ agent ─────────────┐                    ▼
│ policy @ sha        │     ┌──────────────────────────────┐
│ frozen-model @ sha  │────►│  runner  (job → trials)      │
│ vlm @ prompt-hash   │     │  world adapter + verifier    │
│ panel (later)       │     └──────────────┬───────────────┘
└─────────────────────┘                    │
                                           ▼
                            ┌──────────────────────────────┐
                            │ TrialVector                  │
                            │  gates[]  metrics[]          │
                            │  headline = safe_*           │
                            │  optional projection @ ver   │
                            │  audit head (pinnable)       │
                            └──────────────┬───────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
              leaderboard            RL export              hosted eval
              (vector rows)     (trajectory + projection)   (GPU workers)
```

World adapters (the thing Harbor calls Environment, which here is *not* a container):

| `environment.kind` | Implementation | First task |
|---|---|---|
| `lumen-gym` | Gymnasium make of a pinned Lumen env | `lumen-nav-safe` |
| `angiostress-contract` | Frozen prediction vs contract JSON | `angiostress-dias` |
| `frame-source` | `or_audit.media.frames.FrameSource` | synthetic demo, later Cholec80 |
| `lumen-replay` | Replay a captured Lumen episode | dataset generation / SFT |

Agent adapters (the thing Harbor calls Agent):

| `agent.kind` | Input / output | First implementation |
|---|---|---|
| `policy` | obs → action, Gymnasium | random policy, then SB3 checkpoint |
| `frozen-model` | image → mask / structure | AngioStress panel |
| `vlm` | image(s) + instruction → tool calls / answers | later; prompt hash is identity |
| `panel` | human labels → `PerceptionResult` | already exists as `AnnotationBackend` |

---

## 4. Target CLI (Harbor verbs)

P0 implements validate/describe only. Later packages fill the rest. Do not invent a different vocabulary.

```bash
# P0 — this package
or-audit tasks validate docs/examples/tasks/lumen-nav-safe
or-audit tasks describe  docs/examples/tasks/lumen-nav-safe
or-audit datasets validate docs/examples/datasets/lumen-nav-v0

# P1 — first real eval (requires pinned Lumen)
or-audit run -t docs/examples/tasks/lumen-nav-safe -a random --n 30 --out jobs/lumen-nav-safe
or-audit run -d lumen-nav-v0 -a policy@<sha> --n 30

# P3 — jobs, replay, RL export
or-audit run -c job.toml
or-audit replay jobs/lumen-nav-safe --expect-head <hash>
or-audit export-rl jobs/lumen-nav-safe --projection gated_reach_v0 --out rollouts.jsonl

# P4 — registry (public)
or-audit datasets list
or-audit run -d seldinger/lumen-nav@0.1 -a policy@<sha>
```

A job directory mirrors Harbor’s, with the reward file replaced:

```
jobs/<job>/
  config.json
  result.json                 # vector aggregates, never a lone mean-reward
  trial-<task>-<i>/
    config.json
    result.json               # TrialVector + identities
    trajectory.json           # actions, obs, info (and images by ref)
    projection.json           # optional, versioned, for RL only
```

There is no `verifier/reward.txt`. If an RL adapter needs a float, it reads `projection.json`.

---

## 5. First datasets (the Terminal-Bench analog)

Ship small, versioned, claim-bounded. Each dataset is a directory of tasks plus `dataset.toml`.

### D1 — `lumen-nav-v0` (P1, public, `phi=procedural`)

The flagship. Five gym ids already in Lumen, one task each:

| Task | Gym id | Headline |
|---|---|---|
| `lumen-nav-tube` | `Lumen/NavTube-v0` | `safe_success` |
| `lumen-nav-stenotic` | `Lumen/NavStenotic-v0` | `safe_success` |
| `lumen-nav-tortuous` | `Lumen/NavTortuous-v0` | `safe_success` |
| `lumen-nav-safe` (branch) | `Lumen/NavTreeBranch-v0` | `safe_success` |
| `lumen-nav-tortuous-tree` | `Lumen/NavTortuousTree-v0` | `safe_success` |

Gates: `wall_penetration`, `diverged`. Metrics: `raw_success`, `safe_success`, `max_pen`. Oracle: physics. Agent: `policy`. World pin: Lumen commit SHA in every task.

Eval protocol: 30 deterministic episodes, published seeds, replay required before a row is public.

This *is* Lumen’s existing bench, pulled through the harness so a reach-only row cannot publish.

### D2 — `angiostress-v0` (P2, public, `phi=public`)

Wrap the existing DIAS + CathAction contracts as tasks. Agent: `frozen-model`. Oracle: `contract`. Claim footer copied from AngioStress: not clinical validation, not sim-to-real proof. Headline is the contract’s primary metric, plus a calibration/failure-mode vector (the reason that repo exists).

### D3 — `lumen-vision-v0` (P3, public, `phi=procedural`)

Same physics scenes, observation = synthetic fluoro / luminal RGB instead of the 5-D state vector. This is the first *world-model / VLM* surface: the agent must act from images. Lumen already renders these. Headline remains `safe_success`.

### D4 — `cholec-public-v0` (P4, public, `phi=public`)

Cholec80/EndoVis phase and (where labelled) CVS-style tasks via `frame-source`. Agent: `frozen-model` or `vlm`. Oracle: published labels. No named humans. No credentialing report. This is how the existing `perception` vocabulary starts earning its keep as eval, not as privileging.

### D5 — clinical video (explicitly not scheduled)

`phi=deidentified_clinical`. Blocked on V-10, BAA, privacy-office acceptance. Not in the wedge. The task format must already refuse to load it without those fields, so adding it later is a flag not a rewrite.

---

## 6. Work packages

Each package has a ship artifact and an acceptance test. Do the next package only when the previous acceptance is green. Dates are omitted on purpose; the dependency order is the plan.

### P0 — Kernel in this repo (this change)

**Why first:** Harbor’s whole product sits on Task / Dataset / Trial / Job types. We do not have them. Without them every adapter is a one-off script.

**Build:**

- Closed vocabularies: subject, PHI, world kind, oracle kind, agent kind, attestation level.
- `TaskSpec` loaded from a Harbor-shaped directory (`task.toml`, `instruction.md`, optional `verifier.toml`).
- `DatasetSpec` as a list of task paths plus a required headline metric.
- `TrialVector` that cannot collapse to a scalar.
- `ProjectionSpec` as a closed enum (no `eval()` of a rule string).
- Loader invariants: headline ∈ metrics; human determinations refused; `procedural` PHI cannot request attestation; policy/model subjects cannot emit `DecisionRecord`s about people.
- CLI: `tasks validate`, `tasks describe`, `datasets validate`.

**Acceptance:**

- `docs/examples/tasks/lumen-nav-safe` loads and validates.
- `docs/examples/datasets/lumen-nav-v0` validates.
- `float(vector)` raises.
- A task with `emit_human_determination = true` and `subject.kind = policy` is rejected.
- A dataset whose headline is `raw_success` while a `safe_success` metric exists is rejected.
- CI stays green without Lumen or CUDA.

**Not in P0:** talking to Lumen, running a policy, GPU, registry, UI.

### P1 — Lumen adapter (first Harbor-class eval)

**Why second:** it is the only world we own that already has `safe_success`, gym ids, and a bench. It is also PHI-free.

**Build (this repo + a pin to Lumen):**

- Optional extra `or-audit[lumen]` (do not make Newton a hard dependency).
- World adapter: `gymnasium.make(gym_id)` against a **pinned** Lumen commit recorded in `task.toml`.
- Agent adapter: `random` and `gymnasium.Env` policy protocol (`reset`/`step`).
- `or-audit run -t … -a random --n 30` writes a Harbor-shaped job directory.
- Map `info[success|safe_success|unsafe|diverged|max_pen]` onto `TrialVector`.
- Replay: rerun writes an identical vector for the same seed.

**Acceptance:**

- 30-episode job on `lumen-nav-safe` with `random` produces `result.json` containing both raw and safe success.
- Publishing a result that omits `safe_success` is rejected by the writer.
- `or-audit replay` matches the pinned head.
- Documented install: Lumen pin, Newton pin (Lumen already pins Newton).

**Lumen-side (separate PR in that repo, if needed):** none, if gym ids and `info` keys stay stable. If they move, pin breaks loudly — that is the point.

### P2 — AngioStress adapter (first real-data eval)

**Build:**

- World adapter that runs the public contract validators / frozen-model entry points.
- Agent identity = weights hash + code version.
- Scorecard footer is the AngioStress claim boundary, not ours.

**Acceptance:** AngioStress release audit still passes when invoked through `or-audit run`. A result without the claim footer is invalid.

### P3 — Jobs, trajectories, RL export

Harbor’s RL page is: job of trials → reward + token ids. Ours is: job of trials → vector + trajectory + projection.

**Build:**

- `job.toml` / `JobConfig` (agents × tasks × n).
- Trajectory JSON: steps with action, obs (arrays or image refs), `info`, terminated/truncated.
- `export-rl` writes jsonl with `projection` float, episode id, task identity. Closed projection `gated_reach_v0`: `0` if any hard gate failed or `diverged`, else `1` iff `raw_success`.
- Gymnasium vector-env note: training stays in Lumen; *evaluation* of trained checkpoints goes through OR-Audit so the leaderboard cannot be the training reward.

**Acceptance:**

- Train (or load) a trivial policy in Lumen, evaluate it through `or-audit run`, export-rl, and show that a policy with high raw / low safe gets projection `0` on unsafe episodes.
- Trajectory replay reconstitutes the same vector.

### P4 — Registry and public leaderboard

Harbor: `harbor run -d org/dataset@version`. We need the same for `seldinger/lumen-nav@0.1`.

**Build:**

- `registry.json` in a public tasks repo (new, Apache-2.0, like Terminal-Bench; or a `datasets/` tree in Lumen — prefer a dedicated `seldinger-tasks` so the harness and the corpus version independently).
- Leaderboard rows: vector, identities, Lumen pin, n, date. Raw and safe both required.
- CI that regenerates the published `random` and (when we have one) baseline-policy rows.

**Acceptance:** a stranger can `or-audit run -d seldinger/lumen-nav@0.1 -a random` and match the published random baseline within documented noise (seeds are fixed, so noise should be zero).

### P5 — Hosted evals (Prime Intellect analog)

Only after P4 replay is boring.

**Build:** GPU workers that run Lumen jobs; tenant isolation for uploaded checkpoints; no `deidentified_clinical` tasks on the shared pool; audit head stored outside the tenant.

**Acceptance:** two tenants cannot read each other’s weights or results; a PHI-class task is refused on the public pool.

### P6 — Image-conditioned / world-model agents

D3 (`lumen-vision-v0`) plus a VLM agent adapter (prompt + image → action or tool). Identity includes prompt hash and tool schema. This is the “world models and technical medical AI tools” surface the company actually wants. It is P6 because a VLM on a lying verifier is worthless; the verifier has to exist first.

**Acceptance:** a dummy VLM that always inserts blindly fails `safe_success` on stenotic/tree tasks; that failure is the published baseline.

### Explicitly later / never in this plan

- Human-subject determinations, contestation production, privileging UI (`PLAN.md` Phase 0+).
- Intraop, PCCP, autonomous-robot certification.
- Wrapping Harbor as a dependency and stuffing Lumen into a Dockerfile to “be compatible.” Optional *export* of a Harbor task that calls us is fine; it must still store the vector, not only `reward.txt`.

---

## 7. Repository layout (target)

```
or-audit/                         # this repo — the Harbor analog (harness)
  src/or_audit/eval/              # Task, Dataset, TrialVector, loader  (P0)
  src/or_audit/eval/worlds/       # lumen, angiostress, frames          (P1+)
  src/or_audit/eval/agents/       # policy, frozen, vlm                 (P1+)
  src/or_audit/eval/runner.py     # job → trials                        (P1)
  docs/BUILD.md                   # this file
  docs/examples/tasks/            # in-tree fixtures, not the public corpus
  docs/examples/datasets/

seldinger-lumen/                  # worlds (already)
angiostress-benchmark/            # perception contracts (already)
seldinger-tasks/                  # NEW at P4 — public dataset registry (Apache-2.0)
                                  # no patient data; procedural + public only
```

License: harness may stay UNLICENSED until P4; the public tasks repo should be Apache-2.0 like Lumen so labs will actually run it. Do not import proprietary OR-Audit into Lumen.

---

## 8. Testing against this plan

Reuse `ASSESSMENT.md` §6 layers. Mapped onto packages:

| Package | Must be green before it ships |
|---|---|
| P0 | L0 kernel CI (current matrix + new eval-contract tests) |
| P1 | L1 physics: 30-ep deterministic, raw+safe, replay |
| P2 | L2 AngioStress release audit through the runner |
| P3 | Replay + projection never marks unsafe as 1 |
| P4 | Stranger-replay of published rows |
| P5 | Tenant isolation + PHI-class refusal |
| P6 | Image agent cannot pass by ignoring the wall |

Red team from `ASSESSMENT.md` §6.3 still applies, especially verifier gaming (hug `safety_max_pen`) and sim-to-real theatre (a Lumen row marketed as clinical).

---

## 9. Kill / pivot criteria (for B, not for C)

Stop or narrow if:

1. **Nobody runs a task.** After P1 is usable, if no external policy/lab produces a single trial against `lumen-nav-v0` (including us using it on our own policies in public), this is Future A (papers). Stay there on purpose.
2. **The headline gets collapsed.** If customers or our own training loop force a single float that ignores gates, the product has failed its only distinction from a gym wrapper. Kill the leaderboard, do not “just this once.”
3. **Lumen is unpinned.** If evals float on `main` and rows cannot replay, we are a blog. Pin or do not publish.
4. **Clinical FOMO.** If we delay P1–P4 to chase hospital video, we are back on Future C’s kill gates. That is a decision, not a slip.

C-SATS remains the prior for *surgeon scoring*. It is not the prior for this plan.

---

## 10. Immediate next step

P0 is in this repository: the types, the loader, the example task/dataset, the CLI validate path. P1 is the first package that can be demoed like Harbor’s homepage — `or-audit run` on a Lumen env — and should be a follow-on PR against a pinned Lumen commit, not a drive-by import.
