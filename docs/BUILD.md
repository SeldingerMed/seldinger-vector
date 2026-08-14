# BUILD: Harbor for procedural medical AI

**Status:** definitive build plan for Future B (decided).
**Analog:** [Harbor](https://www.harborframework.com/) — evaluate agents in sandboxed environments — specialized for medical world models and technical procedural AI.
**Not:** `PLAN.md` credentialing (Future C). That remains a gated mode on these rails. Do not let hospital ACV block this plan.
**Companions:** [Lumen](https://github.com/SeldingerMed/seldinger-lumen), [AngioStress](https://github.com/SeldingerMed/angiostress-benchmark), [`ASSESSMENT.md`](ASSESSMENT.md).

Harbor’s homepage is one sentence: *evaluate agents in sandboxed environments.* Ours is the same sentence with the sandbox replaced:

> Evaluate medical procedural agents — policies, frozen perception models, VLMs, world models — in physics and image environments, and score them with a vector verifier that cannot hide injury inside a reach metric.

**The evals are infinite. We do not enumerate them.** Harbor does not have a kernel enum for Python vs Rust vs Go. It runs whatever Dockerfile + tests a task author submitted. Same here: we cannot heuristically define CABG next-step, a cath policy, and everything in between. We define a **port** (how the model talks), a **task format** (what the author must bring), and a runner that services `acme/cabg-vlm` the same way it services `seldingermed/cathmodel`.

Lumen is the first *seed world* so the sandbox is runnable. It is not the catalog of medicine.

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

Buyers: anyone training or claiming a technical AI system that *acts in* or *decides from* a procedure. They already need third-party numbers they cannot credibly self-issue. The unit they hand us is `(dataset, agent)`, both `org/name@version` — not a procedure we had to have thought of first.

### 1.1a The sandbox is ports + submissions, not a procedure list (normative)

Harbor’s trick: the **task author** defines the world and the tests. The platform runs them. Harbor does not know what “fix this Python bug” means.

We cannot put an OR in a Dockerfile, and we cannot write a heuristic for every procedure. So the finite set is **how a model talks**, not **what the procedure is called**.

| Finite (kernel) | Infinite (registry) |
|---|---|
| **Ports:** `gym-policy` (obs → action, world steps) and `video-predict` (media → structured prediction) | Procedures, tasks, label schemas, gym ids |
| **World adapters:** gym / frames / contract (how we host a world) | Whose video, which anatomy, which sim |
| **Verifier shape:** vector, abstention, headline rules | Gate ids and metric names the task author declares |
| **Isolation:** PHI class, subject kind, oracle kind | Which hospital, which BAA, which split |
| **Identity:** `org/name@version` for datasets *and* agents | Every model and every bench that will exist |

A new procedure is a published dataset on an existing port. It is not a new enum, not a new company, and not a Seldinger-authored heuristic.

**Service (same verb):**

```bash
or-audit run -d seldingermed/lumen-nav@0  -a seldingermed/cathmodel
or-audit run -d acme/cabg-nextstep@0      -a acme/cabg-vlm
```

`seldingermed/cathmodel` is a `gym-policy` agent. `acme/cabg-vlm` is a `video-predict` agent (next-step / outcome on clips). If someone points the VLM at a gym task, **bind refuses**. We do not invent a CABG adapter to paper over a port mismatch. The kernel does not know CABG.

**Who brings the oracle?** The task author. Labels, a physics `info` dict, or a contract JSON. If they did not bring labels, we cannot score a CABG model — we do not hallucinate anatomy heuristics. That is the honest limit of a sandbox. It is also why this scales: we run and check shape; we do not maintain a medical ontology of every next-step vocabulary.

**What we will not put in the kernel:** a closed list of specialties (endovascular / laparoscopy / …), Strasberg CVS, Cholec80 phases, “suture throw,” or any other named procedure as a load-bearing type. Those may appear as *tags* or as *submitted tasks*. They are not the product taxonomy. Credentialing-mode perception vocabularies stay in `PLAN.md` mode.

Seed fixtures so both ports exist in-tree (not a catalog of medicine):

- `docs/examples/tasks/lumen-nav-safe` — `gym-policy` (Lumen, first runnable in P1)
- `docs/examples/tasks/video-nextstep` — `video-predict` (generic next-step / outcome; procedure is the author’s)
- `docs/examples/agents/seldingermed-cathmodel` — `org/name` on `gym-policy`
- `docs/examples/agents/example-video-predictor` — `org/name` on `video-predict`

Open surgery as a distinct haptic/OR-theatre problem is out of scope for v1 (image-guided start). Knowledge-work clinical evals (Doctronic-class) stay out.

### 1.2 Harbor map (normative)

Harbor’s objects are the right objects. Harbor’s *world* and *reward* are the wrong primitives for medicine. This table is the product. Deviations need a written amendment.

| Harbor | This stack | Keep / change |
|---|---|---|
| `harbor` CLI | `or-audit` CLI | Keep the verbs: `run`, `datasets`, `view` (later) |
| **Task** = instruction + Dockerfile + `tests/` → `reward.txt` | **Task** = `instruction.md` + world spec + vector verifier | Change the world and the verifier |
| **Dataset** = collection of tasks, optional custom metrics | **Dataset** = collection of tasks; custom metrics must not erase the safety vector | Keep |
| **Agent** = Claude Code, OpenHands, Terminus, custom `BaseAgent` | **Agent** = policy checkpoint, frozen model, VLM, (later) panel | Keep the interface, change the population |
| **Environment** = Docker / Daytona / Modal / E2B | **World** = Lumen gym, other Gymnasium sims, FrameSource, AngioStress contract. Containers wrap the *runner*, not the patient | Change |
| **Trial** = one agent attempt; “a rollout that produces a reward” | **Trial** = one agent attempt; a rollout that produces a **vector**. Reward is an optional projection | Change |
| **Job** = cartesian product of agents × tasks × attempts | **Job** = same | Keep |
| Registry of published datasets | Registry of published *procedural* datasets | Keep |
| Cloud sandboxes for 1000s of containers | GPU workers for batched sims; isolated jobs for weights | Change the substrate |
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
6. **Oracle kind is a field.** `physics` (sim contact), `contract` (published labels), `panel` (raters), `script` (Harbor’s `solve.sh`, rare).
7. **Analysis ≠ attestation.** Sim tasks attest nothing. Clinical tasks cannot attest without V-10.
8. **Replay identity** is `(task_id, task_version, agent_identity, seed, world_pin)`. A leaderboard row without this is a blog post.
9. **Port is a field.** Every task and every agent names `gym-policy` or `video-predict`. Bind is port match, not procedure match. A third port is a kernel change with tests, not a tag.

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
                    ┌──────────── datasets (org/name@version) ─────────┐
                    │  seldingermed/lumen-nav   acme/cabg-nextstep  …  │
                    └──────────────────────┬───────────────────────────┘
                                           │ tasks
┌─ agent org/name ────┐                    ▼
│ seldingermed/cathmodel │  ┌──────────────────────────────┐
│ acme/cabg-vlm          │─►│  runner  (bind port, then run)│
│ huggingface/…          │  │  world adapter + verifier    │
└─────────────────────┘     └──────────────┬───────────────┘
                                           ▼
                            ┌──────────────────────────────┐
                            │ TrialVector                  │
                            │  gates[]  metrics[]          │
                            │  headline = task-declared    │
                            │  optional projection @ ver   │
                            │  audit head (pinnable)       │
                            └──────────────┬───────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
              leaderboard            RL export              hosted eval
              (vector rows)     (trajectory + projection)   (GPU workers)
```

World adapters (Harbor’s Environment — substrate, not anatomy):

| `environment.kind` | Implementation | Port it usually serves | Seed |
|---|---|---|---|
| `lumen-gym` | Gymnasium make of a pinned Lumen env | `gym-policy` | `lumen-nav-safe` |
| `gym` | Any other Gymnasium env the task names | `gym-policy` | none in-tree; third-party gym_id |
| `frame-source` | `or_audit.media.frames.FrameSource` | `video-predict` | `video-nextstep` |
| `angiostress-contract` | Frozen prediction vs contract JSON | `video-predict` | P2 |
| `lumen-replay` | Replay a captured Lumen episode | either | dataset generation / SFT |

Agent packages (Harbor’s Agent — identity is `org/name`):

| `port` | Input / output | Seed agent | First runner |
|---|---|---|---|
| `gym-policy` | obs → action | `seldingermed/cathmodel` | P1 |
| `video-predict` | clip → JSON the task named | `example/video-predictor` | P2 (AngioStress as first real dataset on this port) |

---

## 4. Target CLI (Harbor verbs)

P0 implements validate/describe only. Later packages fill the rest. Do not invent a different vocabulary.

```bash
# P0 — this package
or-audit tasks validate docs/examples/tasks/lumen-nav-safe
or-audit tasks validate docs/examples/tasks/video-nextstep
or-audit agents validate docs/examples/agents/seldingermed-cathmodel
or-audit bind docs/examples/tasks/lumen-nav-safe \
              docs/examples/agents/seldingermed-cathmodel
or-audit datasets validate docs/examples/datasets/lumen-nav-v0

# P1 — first real gym-policy eval (requires pinned Lumen)
or-audit run -t docs/examples/tasks/lumen-nav-safe -a seldingermed/cathmodel --n 30
or-audit run -d seldingermed/lumen-nav@0 -a seldingermed/cathmodel --n 30

# P2 — first real video-predict eval (AngioStress, then anyone's corpus)
or-audit run -d seldingermed/angiostress@0 -a seldingermed/cath-seg
or-audit run -d acme/cabg-nextstep@0 -a acme/cabg-vlm

# P3 — jobs, replay, RL export
or-audit run -c job.toml
or-audit replay jobs/lumen-nav-safe --expect-head <hash>
or-audit export-rl jobs/lumen-nav-safe --projection gated_reach_v0 --out rollouts.jsonl

# P4 — registry (public): datasets *and* agents
or-audit datasets list
or-audit agents list
or-audit run -d acme/cabg-nextstep@0 -a acme/cabg-vlm
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

## 5. Datasets (the Terminal-Bench analog)

The registry is **submitted procedural evals**, not a list of specialties we thought of. D1 and D2 are seeds so both ports have a first-party bench. Everything else is `org/name@version` that someone else published.

Ship small, versioned, claim-bounded. Each dataset is a directory of tasks plus `dataset.toml`. Tasks in a dataset share a headline and a PHI class. They do not share a procedure name — the kernel has none.

### D1 — `seldingermed/lumen-nav` (P1, public, `phi=procedural`, port=`gym-policy`)

**First runnable dataset, not the flagship of the company.** Five gym ids already in Lumen, one task each. Headline `safe_success`. Physics oracle. Agent: any `gym-policy` package (`seldingermed/cathmodel`, `random`, a stranger’s checkpoint).

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
- `seldingermed/cathmodel` binds to the gym task and is refused on the video task.
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
- `or-audit run -d seldingermed/lumen-nav@0 -a seldingermed/cathmodel --n 30` writes a Harbor-shaped job directory.
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

**Acceptance:** two tenants cannot read each other’s weights or results; a PHI-class task is refused on the public pool; `acme/cabg-vlm` against `acme/cabg-nextstep` is the same job type as `seldingermed/cathmodel` against `seldingermed/lumen-nav`.

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

## 10. Immediate next step

P0 is in this repository: the types, the loader, both port seeds, `org/name` agents, bind, the CLI validate path. P1 is the first *runnable* gym-policy package. P2 is the first *runnable* video-predict package — the path that services an uploaded next-step model. Neither may require the other port’s fixture to be deleted.
