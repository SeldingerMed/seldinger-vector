# OR-Audit as Harbor for medicine

**Status:** v0.3 implementation assessment; [`BUILD.md`](BUILD.md) is the product architecture and [`V0.3.md`](V0.3.md) is the migration record.
**Scope:** architectural rationale, hostile readings, evidence boundaries, and pre-deployment test layers.
**Companions:** [Seldinger Lumen](https://github.com/SeldingerMed/seldinger-lumen), [AngioStress](https://github.com/SeldingerMed/angiostress-benchmark), and the in-tree counterfactual world-model path.

This document exists because `PLAN.md` and the founding intent are not the same product.

- `PLAN.md` is a **credentialing / QA attestation** thesis: sell a vendor-neutral score of *named surgeons* to hospitals and robot-platform challengers, under a legal wrapper a risk officer can hold.
- The implemented product is an **evaluation and assessment layer for technical AI in medicine**: physical AI, image-conditioned decisions, world models, and robotics. Agents bind to task interfaces by declared capability, not by specialty or a closed port enum.

**Decision (2026-08-14): Future B is the wedge.** Build Harbor-for-medicine — niche eval / RL infrastructure for procedural medical AI and world models. Named-human credentialing (`PLAN.md`) stays a gated mode on the same rails and does not block this plan.

The definitive build spec is [`BUILD.md`](BUILD.md). This file remains the straw-man / steel-man and the pre-deployment test layers.

v0.3 completes the framework layer this assessment originally identified as missing: versioned task/taskset/agent contracts, capability satisfaction, isolated package runtimes, closed-loop/interactive/single-turn/counterfactual harnesses, typed procedural traces, vector scorecards, replay, and declarative RL projections.

The remaining questions in this assessment concern external adoption, task quality, data rights, calibration, and deployment—not whether the shared harness contract exists.

---

## 1. What actually exists

The common v0.3 contract now connects the OR-Audit verifier, Lumen worlds, AngioStress contracts, and counterfactual model evaluation.

### 1.1 This repo (`or-audit` v0.3.0a0)

The repository now contains two explicit layers:

| Layer | Load-bearing implementation |
|---|---|
| `eval.contracts` | Interfaces, capabilities, harness modes, scenarios, perturbations, runtime descriptors |
| `eval.loader` | Canonical v0.3 packages plus deterministic v0.2 normalization |
| `eval.bind` | Schema-level capability satisfaction and accepted agent-kind checks |
| `eval.plugins` / `plugin_host` | Persistent JSON subprocess protocol; explicit trusted in-process test mode |
| `eval.runner` | Closed-loop, interactive, single-turn, and counterfactual dispatch without procedure branches |
| `eval.trace` | Typed procedural evidence for transitions, uncertainty, failure, recovery, handoff, tools, and timing |
| `eval.vector` / `scorecard` | Hard gates plus boolean, continuous, categorical, and unassessable metric semantics |
| `eval.job` / `reconstitute` | Portable package bundles, content heads, vector reconstruction, deterministic replay |
| `eval.registry` | Taskset and agent resolution with v0.2 dataset compatibility |
| Existing domain/audit/de-id/scoring modules | Credentialing-mode invariants and evidence controls retained beside the eval framework |

Three in-tree executions cover the generalized surface: Lumen closed-loop policy evaluation, procedural-video structured prediction with abstention, and counterfactual consequence ranking with uncertainty and recovery events.

### 1.2 Lumen (open core, Layer 0)

A differentiable, GPU-parallel solver for *a continuum instrument in a deformable lumen, observed through a sensor*. Invariants that matter for this assessment:

- Core names no anatomy. Repurposing across endovascular / endoscopic / GI is a profile swap, not a fork.
- Two tiers: fast (Newton VBD, RL throughput) and accurate (IPC reference, autodiff calibration).
- Gymnasium envs (`Lumen/NavStenotic-v0` and kin). Action is insertion + twist. Info already splits **`success` from `safe_success`** — raw target-reach versus wall-safe reach.
- Open/closed firewall: no patient data in the public repo; `provenance="procedural"` on committed assets; real-data calibration stays private.
- Dataset machinery: capture, validate, index, split, synthetic fluoro, luminal RGB, CV labels.

Lumen is already an *environment* in the Harbor sense, except the world is physics rather than a Docker filesystem, and the verifier is contact/penetration rather than `tests/test.sh`.

### 1.3 AngioStress (real-data perception benchmark)

A frozen-model stress test on real DSA (DIAS) and endovascular segmentation (CathAction). Contract-validated, release-audited, claim-bounded: it is a measurement package, not clinical validation and not proof of sim-to-real transfer.

AngioStress is already a *dataset + verifier* in the Harbor sense, except the agent is a frozen segmentation model rather than a coding agent, and the sandbox is a prediction contract rather than a container.

### 1.4 The framework piece is now implemented

OR-Audit's unit of work is a versioned task package:

```text
task.toml       interface, harness, pinned world, metric and gate declarations
instruction.md  agent-visible task instruction
inputs/world    observations or task inputs
labels/oracle   verifier-only evidence
verifier.py     task-owned vector scoring
```

An agent package declares capabilities and runtime identity. `bind` proves compatibility before execution. `run` invokes the matching generic harness, keeps labels out of the agent request, and emits a typed trace plus vector. `replay` reconstructs the vector through the bundled verifier and checks package and artifact heads. `export-rl` applies only the task's declarative projection rule.

The missing product evidence has moved outward: third-party task authors, external model teams running unchanged packages, calibrated real-data tasks, and buyers using the artifacts in an actual assurance or training workflow.

---

## 2. Straw man

A straw man is useful when it is the argument a sophisticated outsider will actually make, not a cartoon. These are the arguments that should be assumed true until a gate falsifies them.

### 2.1 Against OR-Audit-as-credentialing (`PLAN.md` as written)

1. **C-SATS already ran this play.** Credible people built video-based GEARS scoring, J&J bought it, it became education inside a manufacturer. The independent attestation body did not emerge. `PLAN.md` §4 names this; a straw man just believes the prior.

2. **The incumbent now ships the feature for free.** My Intuitive+ / Case Insights is AI skill scoring on da Vinci 5. Selling "we also score surgery" is selling a worse version of a bundled feature. Neutrality is a preference, not a budget line.

3. **Credentialing is a mandate with no software budget.** Today's spend is attending time, often vendor-subsidized. An LOI from a friendly program is not an ACV. `PLAN.md` V-4 is still open; until an unsubsidized paying partner exists, this is a services shop.

4. **The artifact is legally radioactive.** A durable finding that surgeon X failed CVS is discoverable. If PSQIA/peer-review does not cover credentialing use (V-3), the buyer is a risk officer whose job is to not hold this file. Accuracy is irrelevant.

5. **v0.1 is type-system theatre.** Frozen pydantic models that raise on `__float__` are not a perception system. The "platform" redacts synthetic 16-pixel overlays on 64×96 arrays, then scores expert annotations the caller already supplied. The interesting work is the work that is not here: models, data rights, raters, counsel.

6. **Annotation economics are a treadmill.** Expert surgeons labelling Strasberg criteria at clinical rates will dominate COGS. The flywheel is a cost center until unit cost is measured (V-7). Harbor-class businesses work because unit tests are cheap. Medical labels are not.

7. **Niche floor.** Robot-platform challengers plus privileging committees plus QA departments may not be a hundred buyers. Below that, the honest exit is a tuck-in, which is C-SATS.

### 2.2 Against OR-Audit-as-Harbor-for-medicine

1. **Cargo-culting Terminal-Bench onto the OR.** Harbor's isolation primitive is a container. A procedure's isolation primitive is PHI, radiation, sterility, and a patient. Putting `instruction.md` in front of a cholecystectomy video does not make it an environment.

2. **`reward.txt` in [0, 1] is the bug, not the feature.** Harbor verifiers write a scalar. `PLAN.md` §7.1 exists because averaging CVS failure into GEARS efficiency is how you mint false confidence. Porting Harbor naively reintroduces the collapse this repo spent its alpha forbidding.

3. **Labs will not train on PHI, and sim is not the task.** Lumen is procedural geometry. Policies that look good on `nav_tree_branch` will fail on real fluoro (AngioStress exists specifically because this is true). An eval hub of sims is a leaderboard for overfitting.

4. **Who pays for medical eval environments?** Prime Intellect's hub feeds Prime Intellect's training. Harbor's customers are model labs and agent-framework authors. The medical-AI lab count is small, the ones with budget already have internal benches, and the ones without budget will clone Lumen (Apache-2.0) and ignore the attestation kernel.

5. **The stack is three repos that do not talk.** Lumen's `Episode` is not `or_audit.domain.Episode`. AngioStress's contract JSON is not a `DecisionRecord`. A "Harbor for medicine" announcement without a shared task schema is a slide.

6. **World models and VLMs are a different competence.** This repo's authors have been unusually careful about gates, hashes, and abstention. That competence is not the same as training a video-action policy or a fluoro world model. Building Harbor-the-harness does not produce Harbor-the-frontier-model, and pretending it does is how a tools company dies in a model company's costume.

### 2.3 Against the company, independent of product shape

Seldinger is nascent. Public repos are research artifacts (Lumen, AngioStress, GaugeFlow, failure-mode preprints). `or-audit` is private, eight commits, UNLICENSED, pre-Phase-0. The steel-man below does not get to skip this. A kernel can be right and still be too early, too narrow, or too expensive to be a company.

---

## 3. Steel man

A steel man is the strongest version of the thesis that is still consistent with the code and with `PLAN.md`'s own kill criteria.

### 3.1 What v0.1 actually got right

Harbor assumes a world that can be copied into a sandbox, a task that is solved or not, a verifier that is a test suite, and a result that is a number. **None of those are true for technical AI in medicine.** The work that looks like over-engineering in a coding-agent harness is the minimum viable contract here:

1. **Isolation is de-identification, not a container.** Harbor's `environment/Dockerfile` keeps the agent away from the host. OR-Audit's `MediaAsset.require_readable` keeps perception away from PHI. The overlay-attestation rule (analysis ≠ attestation; V-10 must be a measurement, not a comment) is the medical equivalent of "the verifier cannot see the solution." A Harbor port that skips this will leak patients. A credentialing product that skips this will not get past a privacy office.

2. **The verifier must abstain.** Harbor tasks that cannot decide fail the agent. Medical tasks that cannot see must not fail the *subject* — whether the subject is a surgeon or a policy. `Determination.INDETERMINATE` and `GateStatus.NOT_ASSESSABLE` are the difference between an eval and a smear. Coding-agent benches do not need this. Procedural benches do.

3. **Safety is not a reward weight.** Lumen already learned this at the physics layer (`success` vs `safe_success`; CathSim comparison: 100% raw reach, 6.7% safe). OR-Audit learned it at the attestation layer (`SafetyGateSet` cannot be averaged). A Harbor-for-medicine that writes one float to `reward.txt` would undo both.

4. **Ground truth is a panel, and the panel is fallible.** Harbor oracles are `solve.sh`. Medical oracles are raters who disagree. ICC(2,1), relative targets, stratified cohorts, and a panel-adequacy floor are how you stop a leaderboard from ranking models against noise. AngioStress's frozen-model, contract-validated posture is the same idea on perception.

5. **The result is an artifact with a chain of versions, not a leaderboard cell.** Model labs will eventually need this anyway (PCCP, CHAI-style model cards, assurance). Building digest-pinned, rule-versioned, contestable outputs in v0.1 is early for a startup and late for a regulated domain.

6. **Video-first, kinematics-optional is the only cross-platform stance that does not require permission from Intuitive.** That argument in `PLAN.md` §7 survives even if the buyer is a lab rather than a hospital: a bench that requires da Vinci telemetry is a bench Intuitive controls.

The steel man of v0.1 is therefore not "we can score robotic surgery." It is: **the eval harness for procedural AI has extra load-bearing constraints that Terminal-Bench never had, and those constraints are now encoded so they cannot be casually deleted.**

### 3.2 What Lumen actually got right

Lumen is the environment Harbor cannot express in a Dockerfile.

- The world is a contact problem, not a filesystem. Agents act with insertion/twist (and, later, more), and the world answers with images (fluoro, RGB) plus safety state (penetration, force).
- Differentiability and GPU batching make it an RL environment, not a demo.
- The firewall (no patient data, no CathSim license contamination) is the open-core analogue of OR-Audit's de-id gate.
- Splitting raw success from safe success is the physics version of OR-Audit's vector score. Any future shared task schema should treat that split as sacred.

A Harbor-for-medicine that does not have Lumen is a video-QA benchmark. A Lumen that does not have OR-Audit is a gym wrapper whose leaderboard will, under competitive pressure, optimize reach and ignore wall injury. Together they are a procedural eval stack. Apart they are papers.

### 3.3 The actual thesis, restated

Harbor/Prime Intellect for medicine is not "run Claude Code on surgical videos." It is:

> A versioned registry of **procedural environments** (sim, phantom, public video, de-identified clinical video) in which **agents** (policies, VLMs, frozen perception models, human raters, later robots) produce **trajectories**, which a **vector verifier** (hard safety gates + soft skill/task metrics + required abstention) scores into an **attested artifact** that is reproducible, contestable, and — if and only if the legal gates clear — holdable about a named person or a named model.

That sentence fits all three repos. It does not require hospital privileging to be the first dollar. It does not throw away `PLAN.md`'s legal work: named-human scoring is a *mode* of the same harness, with extra gates, not a different codebase.

The C-SATS prior kills **independent surgeon-scoring as a venture-scale wedge**. It does not kill **third-party eval infrastructure for procedural models**, because that market did not exist in 2018. Native vendor scoring (`PLAN.md` §4) is then the same timing argument, pointed at a different buyer: as Intuitive, Medtronic, and the labs all ship numbers, an independent, vector-valued, abstention-capable, sim-plus-real bench becomes the thing they cannot credibly self-issue. That is `PLAN.md`'s neutrality claim, aimed at models rather than surgeons. The surgeon product remains an option on the same rails.

---

## 4. What this could become

Three futures, in increasing ambition. Only the first two are available without pretending Phase 0 is done. The third is `PLAN.md`'s original wedge, demoted from "the product" to "a mode."

### Future A — Research harness (default if no commercial motion)

Keep OR-Audit private, keep Lumen/AngioStress public, publish tasks as papers. Useful, cheap, not a company. The steel-man kernel still pays off as citation and as internal discipline. Kill criterion: if after the refinement below there is still no external user of a task contract, stay here on purpose rather than by neglect.

### Future B — Procedural eval hub (recommended near-term product)

The Harbor analog.

**Buyers, in likely order of willingness to pay without a hospital committee:**

1. **Robot-platform and policy teams** (Lumen users, endovascular/endoscopic autonomy groups). They already need `safe_success`, sim-to-real stress, and a third party who will not let them hide wall injury inside a reach metric. This is `PLAN.md` Segment 1, minus the fiction that the artifact is about a surgeon.
2. **Medical-AI / CV labs** (AngioStress users, Theator-class, academic groups). They need frozen, versioned, claim-bounded evaluation. `PLAN.md` Segment 4, sold as *evaluation*, not as labelled ground truth — which avoids the moat-leak `PLAN.md` §10 warned about.
3. **Assurance / CHAI-class / future PCCP work.** Long-horizon, same artifacts.

**What is sold:** hosted or licensed eval runs against versioned tasks; attested scorecards for *models and policies*; optional private tasks. Rubrics stay public. The corpus, the panel, and the chain are the product.

**What is not sold in this future:** determinations about named clinicians. That keeps V-2/V-3 from blocking the first ship. De-id and audit still ship, because even model-eval on clinical video is PHI.

### Future C — Credentialing / QA attestation (`PLAN.md` as written)

Still real, still gated on Phase 0 (demand, legal holdability, annotation economics, data rights, C-SATS answer). Reuses Future B's harness. Adds named-human mode, privilege posture, contestation in production, committee UX.

Do not build C's commercial motion until B has a runner and at least one external task user. Do not build C's legal motion in parallel as if it were free: counsel for V-2/V-3 is still worth a Phase 0 desk opinion, because B-on-clinical-video will hit adjacent questions. But the *company-killing* unsubsidized-hospital-ACV gate should not block B.

### Endgame (option, not a plan)

Certification of assistive/autonomous procedural systems. Same as `PLAN.md` §11: gated on corpus, partners, regulatory clarity, and pull. Lumen + attested evals are the compounding assets *if* B works. They are not a reason to skip B.

---

## 5. Refinement plan

Refinement is a change to the *product shape* of this repository so that Future B is expressible without deleting Future C. It is not a rewrite of Lumen and not a claim that Phase 0 is cleared.

### R0 — Re-thesis in this repo (this change)

- Treat `PLAN.md` as the credentialing-mode spec and this document as the eval-harness spec. If they conflict on a kernel invariant (no scalar collapse, abstention, de-id as gate, video-first, audit chain), the kernel wins.
- Stop describing the demo as a credentialing product in the README lead. It is a harness demo that can emit a credentialing-shaped report.
- Name the non-goal precisely: **not** "we are not Harbor." **Yes** "we are not Harbor-the-Docker-runner, and we will not write `reward.txt`."

### R1 — A task contract that all three repos can implement

Introduce a Harbor-shaped but medically honest unit of work. Sketch (normative intent; not yet code):

```
tasks/<id>/
  task.toml              # id, version, modality, agent_kinds, resources
  instruction.md         # what the *agent* is told (never the gold)
  environment.toml       # which world: lumen-gym | angiostress | video-episode | phantom
  verifier.toml          # gate set + metrics; vector, not scalar
  solution/              # optional oracle (expert annotations or scripted policy)
```

Required fields Harbor does not have:

| Field | Why |
|---|---|
| `safety_vector` | Hard gates reported separately from task metrics |
| `abstain_ok` | Unassessable is a legal outcome, not a runner error |
| `phi_class` | `procedural` / `public` / `deidentified_clinical` / `prohibited` |
| `subject_kind` | `model` / `policy` / `human` — human unlocks contestation + privilege gates |
| `oracle_kind` | `script` / `panel` / `physics` — panel implies ICC machinery |
| `attestation` | `none` / `analysis_only` / `attested` — same split de-id already enforces |

Adapters, in order, so the contract is proven on things that already exist:

1. **Lumen gym task** — wrap `safe_success` / `success` / `unsafe` / `diverged` as a `SafetyGateSet` analogue. No PHI. This is the first Harbor-class task and should be runnable from this repo against a pinned Lumen commit.
2. **AngioStress contract task** — wrap the v0.1 real-data contract as a frozen-model eval. Claim boundary copied verbatim.
3. **Synthetic video task** — today's demo, re-expressed as a task, so the existing pipeline is a backend of the runner rather than the product.
4. **Public surgical video** (Cholec80 / EndoVis) — perception-only, no named humans, no credentialing report.
5. **De-identified clinical video** — blocked on V-10 measurement + a privacy-office acceptance; this is where Future C begins to share rails with B.

Do not add a Docker-shaped `environment/Dockerfile` as the primary world. Lumen is the world for sim; a `FrameSource` is the world for video. Containers may wrap the *runner*, not the patient.

### R2 — An agent protocol next to `PerceptionBackend`

v0.1 already has the right seam: everything above perception consumes `PerceptionResult`. Extend, do not replace:

- `AnnotationBackend` (exists) — human / panel
- `FrozenModelBackend` — AngioStress-style, identity is weights hash + code version
- `PolicyBackend` — Lumen policy, identity is checkpoint + action space version
- `VlmBackend` — image-conditioned decisions, identity is model + prompt hash + tool schema
- `OraclePolicyBackend` — `solution/` for solvability checks (Harbor's `solve.sh`)

Every backend must declare `observes` (already required). A backend that does not look at bleeding cannot clear the bleeding gate. That rule is the whole product.

### R3 — A runner that emits a vector artifact

`harbor run -d dataset -a agent -m model` analogue:

```
or-audit run --task tasks/lumen-nav-safe --agent policy@sha --n 30 --out runs/...
```

Outputs, all required:

- per-episode score vector (gates + metrics)
- abstention rate
- determination *only if* `subject_kind` allows it and a `DecisionRule` was pre-registered
- audit log with externally pinable head
- a rendered scorecard that cannot print a composite scalar

Explicitly **do not** produce `reward.txt` as the primary interface. A scalar may be derived for RL *behind* a documented projection (`safe_success` as 0/1, or a gated reward that is zero on any hard-gate fail), and that projection is itself versioned. This is how Lumen's PPO training and OR-Audit's attestation stay compatible without lying.

### R4 — Open the kernel, keep the corpus

Match Lumen's firewall:

- Task schemas, verifier code, synthetic/procedural tasks: open or open-core.
- Clinical video, rater identities, named-human artifacts: private, never in this git history.
- Field-of-use on any label license (`PLAN.md` §10) still applies if labels are sold. Selling *eval runs* is the default so labels need not leave the building.

UNLICENSED is fine until R1 exists. After R1, pick a license on purpose (kernel Apache-2.0 like Lumen, or keep proprietary if the harness *is* the company). Do not accidentally contaminate Lumen with a proprietary import in the other direction.

### R5 — Product surfaces, in ship order

1. **CLI runner + one Lumen task + one AngioStress task + the existing synthetic demo as a task.** Internal users only. This is "v0.2," still alpha.
2. **Public leaderboard for procedural-sim tasks** (safe success, wall metrics, abstention, hashes). Lumen already has benchmark snapshots; stop leaving them as README numbers.
3. **Hosted evals** (Prime Intellect analog) only after the runner is deterministic given `(task_version, agent_identity, seed)`.
4. **Human-subject mode** only after `PLAN.md` Phase 0 legal + demand gates. The code path can exist earlier behind a flag that refuses to emit `subject_kind=human` artifacts.

### R6 — What not to refine yet

- Intraoperative decision support, live gating, robot certification (`PLAN.md` §11).
- Kinematics as a required signal.
- A GEARS-first headline.
- Selling labelled surgical video to labs as the business.
- A UI for privileging committees.
- Wrapping Harbor/Verifiers as a dependency and hoping medical constraints fit in `reward.txt`.

---

## 6. Testing before deployment

"Deployment" is not one event. Shipping a Lumen leaderboard, a hosted model eval, and a named-surgeon determination are three different products with three different bars. The mistake would be to treat the current 90% unit-coverage floor as the bar for any of them.

### 6.0 What v0.1 already tests (keep, do not dilute)

These are the right tests for a kernel. They are the wrong tests for a perception system.

- Domain invariants (video required, attestation digest matches status, timezone-aware episodes).
- De-id *leak* regressions: stride gaps, short exits, overlay thinner than the grid, attestation without measurement, hashing the writer's bytes not the caller's.
- Gate evidence discipline: low confidence cannot PASS or FAIL; undeclared observation kinds cannot clear; binding to the wrong media raises.
- Agreement-gate protections: mixed-band cannot headline; GEARS-only cannot pass; panel ICC below adequacy cannot pass; relative target cannot be driven to zero.
- Audit: canonical stability, chain verify, unpinned verify does not exit 0.
- Composition: synthetic demo produces determinations that track scores, withheld attribution when privilege is unconfirmed, intact chain.

**Rule:** any refinement that makes one of these tests weaker is a product regression, even if it makes a demo prettier.

### 6.1 Test layers (must be named, because "we have tests" will be used to ship)

| Layer | Object under test | Oracle | Ships with |
|---|---|---|---|
| L0 Kernel | types, gates, hashes, de-id policy | synthetic fixtures, adversarial cases | every commit (current CI) |
| L1 Physics env | Lumen tasks, contact, `safe_success` | physics (penetration, target distance), pinned seeds | Lumen CI + OR-Audit adapter CI |
| L2 Perception bench | frozen models on public/real-data contracts | AngioStress-style contracts, not "looks good" | AngioStress release audit |
| L3 Public video | phase / CVS / bleeding on Cholec80/EndoVis | published labels + a held-out split | research only, claim-bounded |
| L4 De-id on real capture | overlay/out-of-body on *in-scope hardware* | privacy-office sample review + measured min identifier px (V-10) | any attested clinical video |
| L5 Panel | human raters vs each other and vs automation | ICC(2,1), Fleiss κ, stratified cohort (`PLAN.md` §13) | any automated skill/safety claim |
| L6 Shadow | partner video, no determinations issued | disagreement review, contestation dry-run | before any human-subject artifact |
| L7 Legal tabletop | privilege, retention, discovery, intended use | counsel memo (V-2, V-3, V-5) | before Future C, and before B on identifiable clinical video |
| L8 Hosted runner | determinism, isolation, pin of audit head | bit-identical replay on `(task, agent, seed)` | before any public leaderboard number |

L0 is green today. L1–L2 are reachable without a hospital. L4–L8 are the actual pre-deployment work. Skipping from L0 to "customers" is how this becomes the straw man.

### 6.2 Pre-deployment gates, by surface

**Surface B1 — public sim leaderboard (Lumen tasks via OR-Audit runner)**

Deploy only if:

- Adapter pins a Lumen commit and a task version.
- `success` and `safe_success` are both reported; a run that reports only reach is rejected by the runner.
- Diverged / NaN episodes are terminal failures, not silent zeros (Lumen already does this; the adapter must not undo it).
- Replay of a published row regenerates the same vector.
- No clinical media, no human subjects, no attestation language that sounds like FDA.

**Surface B2 — frozen perception eval (AngioStress via the same runner)**

Deploy only if:

- AngioStress release audit still passes.
- Claim boundary is copied into the scorecard footer (not clinical validation, not sim-to-real proof).
- Weights identity is a hash, not a marketing name.
- Test / train contamination checks exist for every dataset.

**Surface B3 — hosted evals for external labs**

Deploy only if B1+B2 gates hold, plus:

- Tenant isolation (their weights never land in our training set; our private labels never land in their checkout).
- PHI class enforced: a `phi_class=deidentified_clinical` task cannot be pulled by a tenant without a BAA path.
- Audit head pinned outside the tenant's reach.
- Rate limits and cost model so annotation-backed tasks cannot be used as a free labelling farm.

**Surface C — named-human credentialing / QA (`PLAN.md` Phase 1)**

Deploy only if `PLAN.md` Phase 0 is actually cleared (three paying design partners including one unsubsidized, counsel on V-2/V-3, measured annotation unit cost, privacy-office acceptance of de-id, written C-SATS answer), **and**:

- L5 panel gate: automated-vs-expert ICC(2,1) ≥ 0.90 × expert-vs-expert on a within-band held-out set, with panel adequacy (`PLAN.md` §13).
- Per-Strasberg-criterion sensitivity at a pre-registered specificity floor, against ≥3-rater consensus, with Fleiss κ reported.
- L4: V-10 discharged *for that capture stack*; default policy still refuses to attest without it.
- Contestability live: access, appeal, disagreement, response, version trail (`PLAN.md` §7.3).
- Determinations cannot be issued when `peer_review_protection_confirmed` is false (already the reporting default).
- Shadow period on real cases with no operational use of the determination.
- Renewal/use gate remains: a score that does not change a credentialing or QA decision is a failed deployment even if L5 is green (`PLAN.md` §12).

### 6.3 Red-team programme (required, not optional)

Harbor's adversarial surface is "the agent cheats the tests." The medical adversarial surface is larger:

1. **PHI escape** — burned-in text below grid resolution, single-frame room flashes, specular highlights that fool redness-ratio, overlays that move, translated/mirrored identifiers, audio if anyone retains it. Extend `tests/test_deid_leaks.py`; then repeat on real capture, not numpy.
2. **Verifier gaming** — a policy that hugs the wall just inside `safety_max_pen`; a VLM that abstains on every hard case to inflate precision; a scorer that averages gates because a customer asked for "one number." The type bans must have runner-level equivalents so an HTTP client cannot do what `__float__` cannot.
3. **Oracle poisoning** — panel drift, one dominant rater, mixed-band evaluation presented as headline, GEARS-only decks. The agreement gate already refuses these; reporting must refuse them too.
4. **Provenance forgery** — swapping media under a digest, truncating the audit tail, rewriting a decision with the same id. Pinned head + length already address truncation; add external pin procedure before B3.
5. **Sim-to-real theatre** — leaderboard trained on Lumen, marketed on clinical stills. AngioStress is the check. Require a real-data row next to every sim row for any external claim that uses the word "endovascular" or "surgical."

### 6.4 What "extensive" means in numbers (provisional, to be derived)

Do not treat these as magic. Derive or replace them the same way `PLAN.md` V-9 requires the 25,000-procedure figure to be derived. Placeholders so the programme is not "we'll know it when we see it":

- **L1:** ≥30 deterministic eval episodes per published Lumen task, plus a throughput/regression job, plus one comparison that still reports the raw/safe split (the CathSim lesson).
- **L2:** AngioStress full-tier surfaces, not the subset.
- **L4:** overlay min-px survey across every capture system in the first deployment, written into `OverlayBoundValidation.source`; human review of a statistically justified sample of attested episodes for residual PHI.
- **L5:** ≥30 within-band cases (`AgreementGate.min_cases`), ≥2 expert raters, preferably 3 for safety labels; power calculation before claiming a sensitivity floor.
- **L6:** enough shadow cases to observe at least one genuine contestation and one genuine `INDETERMINATE`, so those paths are not first exercised in production.

If the budget cannot fund L5, do not ship automated skill scores. Ship L1/L2 only. That is still a product (Future B). It is not Future C.

### 6.5 CI shape after refinement

Keep current lint/type/unit matrix.

Add, as separate workflows so a GPU or data-dependent job cannot gate kernel correctness:

- `adapter-lumen` — optional extra, pinned Lumen, CPU-small task, skipped if Newton is absent.
- `adapter-angiostress` — contract validate + release audit on the public package.
- `replay` — given a checked-in run artifact, regenerate and diff the vector.
- `firewall` — no clinical media, no unexpected licenses (Lumen already has this idea).

Never run L4–L7 in public CI.

---

## 7. Mapping: Harbor / Prime Intellect → this stack

| Harbor / Prime | Naive medical port (reject) | This stack (keep) |
|---|---|---|
| `environment/Dockerfile` | Video files in a container | Lumen gym / FrameSource / AngioStress contract |
| `instruction.md` | "Score this surgeon" | Task instruction to a *model or policy*; humans get a rubric, not a prompt |
| `tests/` → `reward.txt` | One float | Vector: hard gates + metrics + abstention |
| Agent (Claude Code, …) | A VLM in the OR | `PerceptionBackend` / policy / frozen model / panel |
| Oracle `solve.sh` | "The attending's opinion" | Physics oracle (Lumen) or panel with ICC (video) |
| Hosted evals | Upload PHI to a lab cluster | Hosted *sim* first; clinical only under BAA + attested de-id |
| Leaderboard | AUROC on CVS | Stratified, relative-to-panel, raw vs safe, claim-bounded |
| RL rollouts | Reward = reach | Projection of the vector, versioned, zero on hard-gate fail |
| Isolation | Namespaces | De-id gate + firewall + tenant isolation |
| Reproducibility | Image digest | Task version + agent identity + seed + audit head pin |

The rejected column is how this becomes a meme. The keep column is why v0.1's apparent over-engineering is the start of the product rather than a distraction from it.

---

## 8. Decision

Future B is the wedge. See [`BUILD.md`](BUILD.md).

1. **Wedge = evaluation infrastructure.** Task interfaces, isolated runtimes, replay, vectors, and projections ship as one framework.
2. **Harness is the product; task content stays package-owned.** Public and private tasksets use the same contracts.
3. **Three reference paths stay runnable.** Lumen closed loop, procedural-video single turn, and counterfactual world-model ranking prevent the architecture from collapsing back into one benchmark.

v0.3 is the implemented baseline. The next product milestone is one external lab or model team running a published package without a harness change and deciding whether the resulting third-party vector is useful enough to adopt.
