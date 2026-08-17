# BUILD: Harbor for procedural medical AI

**Status:** v0.3 framework implemented; [`V0.3.md`](V0.3.md) is the migration record.
**Analog:** [Harbor](https://www.harborframework.com/) — evaluate agents in isolated environments — specialized for medical world models and technical procedural AI.
**Companions:** [Lumen](https://github.com/SeldingerMed/seldinger-lumen), [AngioStress](https://github.com/SeldingerMed/angiostress-benchmark), [`ASSESSMENT.md`](ASSESSMENT.md).

> Evaluate medical procedural agents — policies, frozen perception models, VLMs, and world models — against pinned task interfaces, and score them with a vector verifier that cannot hide a hard-gate failure inside a reach metric.

The evals remain open-ended. Task authors publish procedure-specific worlds, schemas, scenarios, perturbations, labels, and verifiers. The kernel defines reusable interaction modes and schema-level interfaces instead of procedure names or a permanent list of ports.

v0.3 replaces direct port equality with `InterfaceSpec` requirements satisfied by agent `CapabilitySpec` declarations. The harness dispatches four modes: `closed-loop`, `interactive`, `single-turn`, and `counterfactual`. Agent and verifier package code runs through separate JSON subprocesses by default. All modes write `ProceduralTrace`, typed metric vectors, task-declared projection identities, and replayable package bundles.

Lumen is the first seed world, procedural video is the first structured-prediction surface, and `counterfactual-recovery` is the runnable world-model path. They exercise one kernel rather than defining its taxonomy.

This document records the product architecture. The original P0–P4 build sequence below remains useful history; v0.3 contract names are authoritative where older `port` or `dataset` terminology appears.

---

## 1. Product

### 1.1 What we sell

Niche infrastructure, not a model and not a hospital app.

1. **A task format** for procedural evals: instruction, pinned world, interface, harness, scenarios, perturbations, and vector verifier.
2. **A runner** that turns `(task, agent, seed)` into a replayable typed trial while keeping agent inputs separate from oracle evidence.
3. **Tasksets**: versioned collections of tasks, compatible with v0.2 dataset packages.
4. **An RL interface** that exports trajectories and a task-declared, versioned scalar projection of the vector.
5. **A registry / leaderboard** whose rows retain gates and typed metrics.

Buyers hand OR-Audit `(taskset, agent)` identities as `org/name@version`; binding is decided by interface capability, not by a procedure list.

### 1.1a The sandbox is ports + submissions, not a procedure list (normative)

Harbor’s trick: the **task author** defines the world and the tests. The platform runs them. Harbor does not know what “fix this Python bug” means.

We cannot put an OR in a Dockerfile, and we cannot write a heuristic for every procedure. So the finite set is **how a model talks**, not **what the procedure is called**.

| Finite (kernel) | Infinite (registry) |
|---|---|
| **Interaction modes:** closed loop, interactive, single turn, counterfactual | Procedures, tasks, label schemas, gym ids |
| **Interface shape:** protocol, observation/action/output schemas, required features | Task-authored interface ids and domain vocabularies |
| **World adapters:** gym, frames, contracts, counterfactual state | Whose video, which anatomy, which simulator |
| **Verifier shape:** hard gates, typed metrics, abstention | Gate ids, metric names, units, and categories |
| **Isolation:** PHI class, agent/verifier process boundary, oracle separation | Deployment substrate and institution |
| **Identity:** `org/name@version` for tasksets and agents | Every model and benchmark that will exist |

A new procedure is a published taskset. A new protocol is declared as an interface and capability; a new interaction shape requires one reviewed harness mode.

**Service (same verb):**

```bash
or-audit run -s seldingermed/lumen-nav@0 -a seldingermed/lumen-linear@0
or-audit run -t docs/examples/tasks/video-nextstep -a docs/examples/agents/example-video-predictor
or-audit run -t docs/examples/tasks/counterfactual-recovery \
  -a docs/examples/agents/example-counterfactual-world-model
```

Binding checks interaction mode, protocol version, schemas, required features, and accepted agent kind. The kernel does not invent a procedure adapter to repair an incompatible package.

**Who brings the oracle?** The task author. Labels, a physics `info` dict, or a contract JSON. If they did not bring labels, we cannot score a CABG model — we do not hallucinate anatomy heuristics. That is the honest limit of a sandbox. It is also why this scales: we run and check shape; we do not maintain a medical ontology of every next-step vocabulary.

**What we will not put in the kernel:** a closed list of specialties (endovascular / laparoscopy / …), Strasberg CVS, Cholec80 phases, “suture throw,” or any other named procedure as a load-bearing type. Those may appear as *tags* or as *submitted tasks*. They are not the product taxonomy. Credentialing-mode perception vocabularies stay in `PLAN.md` mode.

Complete in-tree paths:

- `docs/examples/tasks/lumen-nav-safe` + `seldingermed-lumen-linear`: closed-loop policy evaluation.
- `docs/examples/tasks/video-nextstep` + `example-video-predictor`: structured procedural-video reasoning with abstention.
- `docs/examples/tasks/counterfactual-recovery` + `example-counterfactual-world-model`: consequence ranking, uncertainty, failure, and recovery evidence.
- `docs/examples/tasksets/counterfactual-recovery-v1`: canonical v0.3 taskset package.
Open surgery as a distinct haptic/OR-theatre problem is out of scope for v1 (image-guided start). Knowledge-work clinical evals (Doctronic-class) stay out.

### 1.2 Harbor map (normative)

Harbor’s objects are the right objects. Harbor’s *world* and *reward* are the wrong primitives for medicine. This table is the product. Deviations need a written amendment.

| Harbor | This stack | Keep / change |
|---|---|---|
| `harbor` CLI | `or-audit` CLI | Keep `run`, `tasksets`, `bind`, `replay`, and registry verbs |
| **Task** = instruction + Dockerfile + tests → scalar reward | **Task** = instruction + pinned world + interface/harness + vector verifier | Change world, protocol, and verifier |
| **Dataset** = collection of tasks | **Taskset** = versioned collection of tasks; v0.2 dataset input remains readable | Rename canonical contract |
| **Agent** = framework-specific coding agent | **Agent** = capability declarations + pinned runtime identity | Generalize interface |
| **Environment** = container substrate | **World** = Gymnasium, FrameSource, contract, replay, or counterfactual state | Separate runtime substrate from world |
| **Trial** = rollout producing a reward | **Trial** = typed procedural trace producing gates and typed metrics | Keep vector authoritative |
| **Job** = agents × tasks × attempts | **Job** = same | Keep |
| Registry of published datasets | Registry of tasksets and agents | Generalize |
| Cloud sandboxes | JSON subprocess locally; pinned remote/container descriptors represented | Make isolation explicit |
| RL via SkyRL; `reward` + token ids | RL via Gymnasium/SB3/SkyRL; **projection** + trajectory | Change the reward |
| ATIF trajectories (tokens, tools, images) | Procedural trajectories (actions, obs, images, contact, vector) | Keep the idea |
| `harbor view` job browser | Later; JSON artifacts first | Defer UI |

### 1.3 Invariants that Harbor does not have (non-negotiable)

These are already encoded in the credentialing kernel and must hold on every eval path:

1. **Vector, not scalar.** Hard gates and task metrics report separately. `TrialVector` raises on `float`/`int`/`bool`.
2. **Headline is task-owned.** Tasksets name one metric, but scorecards retain every gate and metric.
3. **Abstention is a legal outcome.** Missing evidence is `null` / unassessable, never an implicit pass.
4. **PHI class is explicit.** `procedural` / `public` / `deidentified_clinical` / `prohibited`.
5. **Subject kind is explicit.** `policy` / `model` / `human`; the eval framework does not emit human determinations.
6. **Oracle evidence is separated.** Agents receive observations or task inputs; task-owned verifiers receive labels, physics state, or contract evidence.
7. **Package code is isolated.** Agent and verifier entrypoints use separate subprocesses unless a runtime explicitly declares `trusted-in-process`.
8. **Replay identity** covers task, agent, runtime, world, interface, trace, vector, projection rule, and artifact head.
9. **Binding is schema-level.** A capability satisfies interaction mode, protocol version, observations, actions/outputs, features, and accepted agent kind.

### 1.4 What we will not build

- Docker as the procedural world (Harbor’s sandbox is our *job isolation*, not the procedure).
- `reward.txt` as the primary interface.
- A general coding-agent harness. We do not compete with Harbor on Terminal-Bench.
- Intraoperative decision support, live gating, robot certification.
- Named-surgeon credentialing UX (Future C).
- Selling labelled clinical video as the business.
- A world-model foundation model of our own as a prerequisite. We evaluate other people’s.
- A heuristic catalog of procedures. If we have to name the anatomy to load the task, the sandbox has already failed.

---

## 2. What we already have (so we do not rebuild it)

| Asset | Repo | Maps to Harbor | v0.3 integration |
|---|---|---|---|
| Vector verifier, abstention, audit chain, de-id gate, agreement harness | `or-audit` | Verifier + artifact | Shared by typed eval vectors and existing credentialing mode |
| Gym envs `Lumen/Nav*-v0`, `safe_success`, capture/replay | `seldinger-lumen` | Environment + first tasks | Pinned closed-loop task and policy packages |
| Frozen-model DSA/segmentation contracts, release audit | `angiostress-benchmark` | Taskset + contract oracle | Single-turn package with claim footer |
| Counterfactual recovery fixtures | `or-audit` examples | World-model task + agent | Runnable consequence-ranking path |

The Harbor glue now exists in `or_audit.eval`: contracts, capability binding, isolated runtimes, four harness modes, typed traces, tasksets, vectors, scorecards, registry loading, replay, and RL projection export.

---

## 3. Architecture

```text
taskset org/name@version
        │ tasks
        ▼
TaskSpec.interface ── requirements ──► CapabilitySpec[] ── AgentPackage
        │                                      │
        └──────────── bind satisfies ──────────┘
                         │
                         ▼
              HarnessSpec.interaction_mode
      closed-loop | interactive | single-turn | counterfactual
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      agent subprocess        verifier subprocess
      observations only       oracle evidence
             └───────────┬───────────┘
                         ▼
                  ProceduralTrace
                         +
                  TrialVector
             gates[] + typed metrics[]
                         +
              optional ProjectionSpec
                         ▼
            replay / scorecard / RL export
```

World adapters are substrate, not anatomy:

| `environment.kind` | Interaction mode | Seed |
|---|---|---|
| `lumen-gym`, `gym` | closed loop | `lumen-nav-safe` |
| `frame-source`, `angiostress-contract` | single turn or interactive | `video-nextstep`, `angiostress-dias`, task-authored turn sequences |
| `lumen-replay` | closed loop or single turn | captured evidence |
| `counterfactual` | counterfactual | `counterfactual-recovery` |

Agent packages advertise capabilities:

| Interface seed | Mode | Input → output | Agent |
|---|---|---|---|
| `gym-policy` | closed loop | observation → action | `seldingermed/lumen-linear` |
| `video-predict` | single turn | clip → structured prediction + abstention | `example/video-predictor` |
| `counterfactual-consequence` | counterfactual | state + interventions → consequence ranking + uncertainty | `example/counterfactual-world-model` |

---

## 4. Target CLI (Harbor verbs)

P0–P4 implement validate, bind, run, replay, jobs-as-config, RL export, immutable registry resolution, portable bundles, scorecards, and static vector leaderboards. Do not invent a different vocabulary.

```bash
# P0 — contract
or-audit tasks validate docs/examples/tasks/lumen-nav-safe
or-audit tasks validate docs/examples/tasks/video-nextstep
or-audit agents validate docs/examples/agents/seldingermed-lumen-linear
or-audit bind docs/examples/tasks/lumen-nav-safe \
              docs/examples/agents/seldingermed-lumen-linear
or-audit datasets validate docs/examples/datasets/lumen-nav-v0

# P1 — gym-policy. Pin world_pin in task.toml. CI uses a factory; live Lumen is optional.
or-audit run -t docs/examples/tasks/lumen-nav-safe -a random --n 30 --out jobs/lumen-nav-safe
or-audit replay jobs/lumen-nav-safe --expect-head <hash>

# P2 — video-predict. Same verb; labels vs JSON. AngioStress requires the claim footer.
or-audit run -t docs/examples/tasks/video-nextstep \
         -a docs/examples/agents/example-video-predictor --out jobs/video-nextstep
or-audit run -t docs/examples/tasks/angiostress-dias \
         -a docs/examples/agents/seldingermed-cath-seg --out jobs/angiostress-dias
or-audit run -d docs/examples/datasets/angiostress-v0 \
         -a docs/examples/agents/seldingermed-cath-seg --out jobs/angiostress-v0

# P3 — jobs-as-config, RL export
or-audit run -c job.toml
or-audit export-rl jobs/lumen-nav-safe --projection gated_reach_v0 --out rollouts.jsonl

# P4 — immutable public registry, portable replay, static vector result surfaces
or-audit datasets list
or-audit agents pull seldingermed/lumen-linear@0 --out packages
or-audit run -d seldingermed/lumen-nav@0 -a seldingermed/lumen-linear@0 \
         -n 30 --out jobs/lumen-linear
or-audit replay jobs/lumen-linear/lumen-nav-safe
or-audit leaderboard jobs --out site
```

A job directory mirrors Harbor’s, with the reward file replaced:

```
jobs/<job>/
  bundle/
    task/                    # exact task package
    agent/                   # exact agent package, if not builtin
  config.json               # relative bundle paths + package digests
  bundle.json               # immutable package manifest
  result.json               # vector aggregates, never a lone mean-reward
  scorecard.{json,md,html}  # deterministic human/machine result surfaces
  trial-<task>-<i>/
    result.json             # TrialVector
    trajectory.json         # verifier-reconstitutable evidence
    projection.json         # optional, versioned, for RL only
```

There is no `verifier/reward.txt`. If an RL adapter needs a float, it reads `projection.json`.

---

## 5. Datasets (the Terminal-Bench analog)

The registry is **submitted procedural evals**, not a list of specialties we thought of. D1 and D2 are seeds so both ports have a first-party bench. Everything else is `org/name@version` that someone else published.

Ship small, versioned, claim-bounded. Each dataset is a directory of tasks plus `dataset.toml`. Tasks in a dataset share a headline and a PHI class. They do not share a procedure name — the kernel has none.

### D1 — `seldingermed/lumen-nav` (P1, public, `phi=procedural`, port=`gym-policy`)

**First runnable dataset, not the flagship of the company.** Five gym ids already in Lumen, one task each. Headline `safe_success`. Physics oracle. Agent: any `gym-policy` package (`seldingermed/lumen-linear`, `random`, a stranger’s checkpoint).

This *is* Lumen’s existing bench, pulled through the harness so a reach-only row cannot publish.

### D2 — `seldingermed/angiostress` (P2, public, `phi=public`, port=`video-predict`)

First real-data dataset on the **same port** a CABG next-step model will use. Frozen prediction vs contract JSON. The field schema is AngioStress’s, not a kernel type. Headline is the contract’s primary metric plus a calibration/failure-mode vector. Claim footer copied from AngioStress.

After D2, `acme/cabg-nextstep` is not a new work package. It is a dataset on `video-predict` whose labels Acme (or a third party) brought.

### D3 — `seldingermed/lumen-vision` (P3, public, `phi=procedural`, port=`gym-policy`)

Same physics scenes, observation = synthetic fluoro / luminal RGB. Still `gym-policy` (the world steps); the observation is images. Headline remains `safe_success`.

### D∞ — everyone else’s

A lab publishes `org/dataset` with `task.toml` + labels or a gym pin + a vector verifier. We host the run if ports bind and PHI class is allowed on that pool. We do not pre-register the procedure. Clinical video (`phi=deidentified_clinical`) stays refused on the public pool (V-10, BAA). The task format already carries that field so adding a private pool later is a flag, not a rewrite.

---

## 6. Work packages

Each package has a ship artifact and an acceptance test. Do the next package only when the previous acceptance is green. Dates are omitted on purpose; the dependency order is the plan.

### P0 — Kernel in this repo (this change)

**Why first:** Harbor’s whole product sits on Task / Dataset / Trial / Job types. We do not have them. Without them every adapter is a one-off script.

**Build:**

- Closed vocabularies: subject, PHI, **port**, world kind, oracle kind, agent kind, attestation level.
- `TaskSpec` loaded from a Harbor-shaped directory (`task.toml`, `instruction.md`, optional `verifier.toml`).
- `AgentPackage` loaded from `agent.toml` with HuggingFace-shaped `org/name` identity.
- `assert_bind(task, agent)` refuses port mismatch.
- `DatasetSpec` as a list of task paths plus a required headline metric.
- `TrialVector` that cannot collapse to a scalar.
- `ProjectionSpec` as a closed enum (no `eval()` of a rule string).
- Loader invariants: headline ∈ metrics; human determinations refused; `procedural` PHI cannot request attestation; policy/model subjects cannot emit `DecisionRecord`s about people; **port is required**; video-predict must name a prediction schema (open slug).
- CLI: `tasks validate`, `tasks describe`, `datasets validate`, `agents validate`, `bind`.

**Acceptance:**

- Both port seeds load: `lumen-nav-safe` (`gym-policy`), `video-nextstep` (`video-predict`).
- `seldingermed/lumen-linear` binds to the gym task and is refused on the video task.
- `example/video-predictor` binds to the video task and is refused on the gym task.
- A task without `[port]` is rejected.
- A video-predict task without `prediction` is rejected.
- Tags may say `cabg`; the kernel does not switch on them.
- `docs/examples/datasets/lumen-nav-v0` and `video-nextstep-v0` validate.
- `float(vector)` raises.
- A task with `emit_human_determination = true` and `subject.kind = policy` is rejected.
- A dataset whose headline is `raw_success` while a `safe_success` metric exists is rejected.
- CI stays green without Lumen, a third-party gym, or CUDA.

**Not in P0:** talking to Lumen, running a policy, GPU, registry, UI.

### P1 — Lumen adapter (first *runnable* gym-policy eval)

**Why second:** it is the only world we own that already has `safe_success`, gym ids, and a bench. It is also PHI-free. It is the seed for `gym-policy`, not because the product is a cath simulator.

**Build (this repo + a pin to Lumen):**

- Optional extra `or-audit[lumen]` (do not make Newton a hard dependency).
- World adapter: `gymnasium.make(gym_id)` against a **pinned** Lumen commit recorded in `task.toml`.
- Agent adapter: `random` and `org/name` packages whose port is `gym-policy`.
- `or-audit run -d seldingermed/lumen-nav@0 -a seldingermed/lumen-linear@0 --n 30` writes a Harbor-shaped job directory.
- Map `info[success|safe_success|unsafe|diverged|max_pen]` onto `TrialVector`.
- Replay: rerun writes an identical vector for the same seed.

**Acceptance:**

- 30-episode job on `lumen-nav-safe` with `random` produces `result.json` containing both raw and safe success.
- Publishing a result that omits `safe_success` is rejected by the writer.
- `or-audit replay` matches the pinned head.
- Documented install: Lumen pin, Newton pin (Lumen already pins Newton).
- `or-audit bind` of `example/video-predictor` onto this dataset still fails.

**Lumen-side (separate PR in that repo, if needed):** none, if gym ids and `info` keys stay stable. If they move, pin breaks loudly — that is the point.

### P2 — video-predict adapter (first *runnable* predict eval)

**Why this, not a specialty list:** AngioStress is the first dataset on `video-predict`. A CABG next-step corpus is the same adapter with a different label schema. Building P2 as “endovascular segmentation product” would repeat the catalog mistake.

**Build:**

- World adapter that runs a frozen model against labelled media / contract JSON.
- Agent identity = `org/name` + weights pin + code version.
- Scorecard footer is the dataset’s claim boundary.
- The prediction field names come from the task, not from `or_audit.perception`.

**Acceptance:** AngioStress release audit still passes when invoked through `or-audit run`. A result without the claim footer is invalid. A stranger’s `video-predict` dataset with a different `prediction` slug loads without a harness change (even if we do not yet host their video).

### P3 — Jobs, trajectories, RL export (this change)

Harbor’s RL page is: job of trials → reward + token ids. Ours is: job of trials → vector + trajectory + projection.

**Build:**

- `job.toml` / `JobConfig` (agents × tasks × n). `or-audit run -c job.toml` writes a cartesian parent (`manifest.json` + one Harbor job dir per pair).
- Trajectory JSON: steps with action, obs (arrays or image refs), `info`, terminated/truncated. Replay reconstitutes the vector from that file without trusting a lone stored float.
- `export-rl` writes jsonl with `projection` float, episode id, task identity. Closed projection `gated_reach_v0`: `0` if any hard gate failed or `diverged`, else `1` iff `raw_success`. The float is recomputed from the vector; a stored projection that disagrees is refused. Homemade projection ids are argparse-refused.
- Gymnasium vector-env note: training stays in Lumen; *evaluation* of trained checkpoints goes through OR-Audit so the leaderboard cannot be the training reward.

**Acceptance:**

- Trivial / `random` policy evaluated through `or-audit run` (FakeLumenEnv in CI; live Lumen optional), `export-rl`, and a high-raw / low-safe episode gets projection `0`.
- Trajectory replay reconstitutes the same vector.

### P4 — Registry and public leaderboard

Harbor: `harbor run -d org/dataset@version`. We need the same for datasets *and* agents: `or-audit run -d acme/cabg-nextstep@0 -a acme/cabg-vlm`.

**Build:**

- `registry.json` in a public tasks repo (new, Apache-2.0, like Terminal-Bench; prefer dedicated `seldinger-tasks` so the harness and the corpus version independently).
- Agent registry beside it (`org/name` packages, weights by pin, not by uploading into the harness repo).
- Leaderboard rows: vector, identities, world pin, n, date. Safety-aware headline required when the task declared one.
- CI that regenerates published baseline rows for *seed* datasets only. Third-party datasets are their publishers’ CI.

**Acceptance:** a stranger can publish a `video-predict` dataset we did not author, `or-audit bind` their model, and (PHI-class permitting) `or-audit run` without a harness change. A stranger can `or-audit run -d seldingermed/lumen-nav@0 -a random` and match the published random baseline (seeds fixed, noise zero).

### P5 — Hosted evals (Prime Intellect analog)

Only after P4 replay is boring.

**Build:** GPU workers that run jobs; tenant isolation for uploaded checkpoints; no `deidentified_clinical` tasks on the shared pool; audit head stored outside the tenant. Upload path: agent weights + dataset (or a pointer to a public one). Bind before run.

**Acceptance:** two tenants cannot read each other’s weights or results; a PHI-class task is refused on the public pool; `acme/cabg-vlm` against `acme/cabg-nextstep` is the same job type as `seldingermed/lumen-linear` against `seldingermed/lumen-nav`.

### P6 — Image-conditioned / world-model agents

D3 (`lumen-vision`) plus richer VLM agents on `video-predict` (prompt + clip → the task’s JSON schema). Identity includes prompt hash and schema. This is the “world models and technical medical AI tools” surface. It is P6 because a VLM on a lying verifier is worthless; the verifier has to exist first.

**Acceptance:** a dummy VLM that always inserts blindly fails `safe_success` on gym-policy seeds; a dummy predictor that never abstains fails a video-predict task whose labels include unassessable clips.

### Explicitly later / never in this plan

- Human-subject determinations, contestation production, privileging UI (`PLAN.md` Phase 0+).
- Intraop, PCCP, autonomous-robot certification.
- Wrapping Harbor as a dependency and stuffing Lumen into a Dockerfile to “be compatible.” Optional *export* of a Harbor task that calls us is fine; it must still store the vector, not only `reward.txt`.

---

## 7. Repository layout (target)

```
or-audit/                         # this repo — the Harbor analog (harness)
  src/or_audit/eval/              # Task, Dataset, Agent, bind, TrialVector  (P0)
  src/or_audit/eval/worlds/       # lumen, gym, frames, angiostress         (P1+)
  src/or_audit/eval/agents/       # gym-policy, video-predict runners       (P1+)
  src/or_audit/eval/runner.py     # job → trials                            (P1)
  docs/BUILD.md                   # this file
  docs/examples/tasks/            # seed fixtures, not the public corpus
  docs/examples/datasets/
  docs/examples/agents/           # org/name packages
  docs/examples/jobs/             # cartesian job.toml (P3)

seldinger-lumen/                  # worlds (already)
angiostress-benchmark/            # first real video-predict dataset (already)
seldinger-tasks/                  # NEW at P4 — public dataset+agent registry (Apache-2.0)
                                  # no patient data; procedural + public only
                                  # third parties publish here or against this format
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
3. **A world is unpinned.** If evals float on `main` and rows cannot replay, we are a blog. Pin or do not publish.
4. **Clinical FOMO.** If we delay P1–P4 to chase hospital video, we are back on Future C’s kill gates. That is a decision, not a slip.
5. **We start enumerating procedures.** If the kernel grows a specialty enum, or P1 ships by deleting the video-predict seed, or the public registry a year later can only score models we wrote tasks for, we failed §1.1a. Ports stay two (until a third is justified). Tasks stay submissions.

C-SATS remains the prior for *surgeon scoring*. It is not the prior for this plan.

---

## 10. Current product boundary and next evidence

P1–P4 now run end to end: executable policy and frozen-model packages, task-owned
verifiers, pinned Lumen and AngioStress worlds, self-contained replay bundles,
deterministic scorecards, and the public `org/name@version` registry at
`SeldingerMed/seldinger-tasks`. The next milestone is not another kernel phase.
It is one external lab or model team running a published package without a harness
change and deciding whether the resulting third-party scorecard is worth paying for.
Until that evidence exists, hosted tenancy, accounts, queues, and clinical media stay
deferred.
