# Surgical & Medical Robotics Evaluation + Validation Platform

**Working title:** OR-Audit
**Status:** Implementation plan (v2 — re-thesis after competitive review)
**Scope:** Surgical / procedural / robotic medicine ONLY. Knowledge-work clinical evals (Doctronic-class) are explicitly out of scope — different product, buyer, data rights, modality, and assurance regime. Treat as a separate company decision.

**What changed from v1:** the moat claim moved from *kinematics access* to *vendor-neutral, conflict-free assessment*. v1 named kinematics as the key differentiator; the signal is emitted and controlled by the robot vendors, and at least one vendor now ships native AI skill scoring built on it (§4). v1 also treated incumbent entry as a future risk; it is present-tense. Several external legal and technical claims that v1 asserted are demoted to owned verification items in §V.

---

## 1. Thesis

Automated, objective evaluation of robotic surgical skill and safety is a real problem — robotic credentialing and quality assurance today rely on sparse, subjective expert review.

The wedge is **vendor-neutral automated scoring of robotic surgical video against validated rubrics (GEARS/binary proficiency metrics) and safety standards (Critical View of Safety), delivered under a legal structure that makes the output safe for a hospital to hold.**

Three claims carry the thesis, in order of importance:

1. **Neutrality.** A robot manufacturer's own scoring of surgeon performance *on that manufacturer's robot* is structurally conflicted. A credentialing body, a malpractice defense, or a competing platform cannot treat vendor self-report as independent evidence of competence. Third-party attestation is the product.
2. **Cross-platform comparability.** Native vendor scoring is per-platform and non-comparable. A hospital running da Vinci + Hugo + Versius has no common yardstick, and a challenger platform cannot evidence proficiency parity using the incumbent's scorer or its own.
3. **Legal holdability.** A durable record asserting "surgeon X below benchmark" or "CVS not achieved in case Y" is an adoption blocker unless it sits inside a structure that governs discoverability and use. Solving this is a moat, not overhead (§9).

Kinematics is **enrichment where obtainable, not the thesis.** See §6 and §V-1.

The robotics endgame — becoming the certification/validation layer for autonomous and assistive surgical robots — is a **value hypothesis, not a guaranteed compounding asset.** It is a gated option, pursued only on an explicit, measurable trigger (§11). Nothing here commits capital to it before the trigger fires.

## 2. Problem

- Robotic surgery (Intuitive da Vinci, Medtronic Hugo, CMR Versius, Vicarious, Moon Surgical, Distalmotion) is expanding, and every platform operator needs to credential surgeons and audit quality.
- Skill assessment is largely manual: an attending or proctor watches video and applies a rubric. Slow, subjective, non-scalable, generates no longitudinal data.
- Safety review (e.g., was the Critical View of Safety achieved before dividing the cystic duct) is retrospective and rarely systematic.
- **Assessment is fragmenting along vendor lines.** As each platform ships its own native analytics, a multi-platform hospital accumulates mutually incomparable scores, and no score is independent of the party selling the robot.
- **The record is legally hot.** Surgical video and adverse skill findings are institutionally gated (IRB, data-use agreements, retention policy, discovery exposure). Buyers resist the artifact itself, not just its price.

## 3. Why now

- Robotic platform competition is intensifying beyond Intuitive's near-monopoly; challengers need *independent* objective evidence to win hospital adoption — evidence a vendor cannot credibly self-issue.
- Native vendor scoring shipping (§4) both validates the category and creates the neutrality gap this plan sells into. This is the timing argument: the market is being educated by the incumbent, on terms only a third party can complete.
- Validated instruments already exist (GEARS, OSATS, GOALS, binary proficiency metrics), so the wedge encodes established convention rather than inventing contested ground truth.
- Public video datasets (Cholec80, EndoVis) provide a perception baseline.
- FDA's evolving adaptive-AI framework (Predetermined Change Control Plan, PCCP) is beginning to define how *intraoperative* AI gets validated — a future market this platform is positioned for but does not yet serve.

## 4. Competitive landscape (present tense, not future risk)

v1 treated incumbent entry as a risk to monitor. It has occurred. Plan accordingly.

| Player | What they ship today | Structural limit we sell against |
|---|---|---|
| **Intuitive — My Intuitive+ / Case Insights** | AI evaluation of da Vinci system data, kinematic movement, and video producing "objective skills assessment"; auto-bookmarked video, performance/workflow trends, outlier exploration. Available exclusively with da Vinci 5. ([intuitive.com](https://www.intuitive.com/en-us/digital-solutions/my-intuitive/my-intuitive-plus)) | Single-platform; newest-console-only; **vendor self-report** — conflicted as independent credentialing evidence |
| **C-SATS (Johnson & Johnson, acq. 2018)** | Cloud video upload, expert-rater scoring incl. GEARS, longitudinal skill tracking; folded into J&J Institute education. ([J&J](https://www.jnj.com/media-center/press-releases/johnson-johnson-institute-adds-innovative-analytics-based-learning-platform-to-help-surgeons-improve-technical-skills-and-clinical-outcomes-across-a-range-of-specialties)) | Owned by a device manufacturer — same conflict; positioned as education, not attestation |
| **Theator** | Surgical video → structured case-level intelligence, automated reports | Video analytics, not credentialing attestation; laparoscopic-general focus |
| **Caresyntax** | Whole-OR multi-modal analytics, risk evaluation | OR operations/workflow orientation |

**The C-SATS datapoint is the most important input to §16's niche-floor question and must be treated as evidence, not noise.** The venture-scale version of the v1 wedge existed, was built by credible people, and resolved as a strategic tuck-in inside a device manufacturer rather than an independent platform. Phase 0 must produce an explicit written answer to: *what is materially different now, and is the honest exit a tuck-in?* If the answer is "nothing, and yes," that is a valid and cheap finding.

**Corollary — the competitive risk row inverts.** v1: "if a platform vendor offers this natively, the wedge is non-viable." v2: native vendor scoring is the *precondition* for the neutrality sale, and the wedge is non-viable if instead an **independent, non-manufacturer-owned** attestation body (a specialty society, an assurance lab, a payer consortium) occupies the neutral position first. That is the risk to monitor (§15).

## 5. What this is NOT (non-goals)

- Not an intraoperative decision-support device. Wedge intended use is **education, credentialing, and retrospective QA** — positioned on the non-regulated side of the intended-use line, subject to counsel confirmation (§V-2).
- Not a surgical robot. Not hardware.
- Not a general AI-eval framework (Harbor/Braintrust class). The harness is a delivery vehicle; the product is the **corpus + labels + calibrated scorer + legal wrapper**.
- Not the knowledge-work clinical eval product. Separate.

## 6. Buyers (reordered; no segment claimed as validated pre-Phase 0)

| # | Segment | Status | What they buy | Why they need a *neutral* party |
|---|---------|--------|---------------|--------------------------------|
| 1 | **Robotic platform challengers** (Hugo, Versius, Vicarious, Moon, Distalmotion) | **Hypothesis — sharpest structural need, real budget** | Independent proficiency evidence; surgeon onboarding/training data | Cannot self-certify credibly; cannot use the incumbent's scorer; need parity evidence for hospital procurement |
| 2 | **Robotic credentialing programs / hospital privileging committees** | **Hypothesis — mandatory activity, unproven software budget** | Objective skill scoring, learning-curve tracking, credentialing evidence | Multi-platform comparability; independence from the vendor whose robot is being credentialed on |
| 3 | **Hospitals / quality & risk departments** | **Hypothesis** | Retrospective QA, adverse-event audit, credentialing at scale | Need the record to be defensible and governed (§9) |
| 4 | **Surgical AI / CV startups** (Theator, Caresyntax, Activ class) | **Hypothesis — also competitors; see §8** | Labeled ground truth, perception-model validation | Third-party validation carries weight self-validation does not |
| 5 | Regulators / assurance labs (CHAI-style) | **Long horizon, option** | Standardized validation evidence for adaptive/AI surgical systems | Neutrality is a hard requirement, not a preference |

**Deliberate change from v1:** Segment 2 is no longer labeled "validated demand." Credentialing is a mandatory *activity*, which is not the same as an available *software line item*. Today's spend is largely attending/proctor time, and a material share of the proctoring economy is subsidized by platform vendors. Segment 2's budget existence is a Phase 0 question, not an assumption (§V-4).

Segments 1–2 are the wedge. Segments 3–4 are expansion within the non-regulated zone. Segment 5 is only reachable after the §11 trigger.

## 7. Wedge product spec

**Product:** Independent, cross-platform robotic surgical skill + safety attestation.

**Inputs (ordered by availability, not by richness):**
1. **Surgical video — the required common denominator.** Every platform produces it; it is obtainable without vendor cooperation via the hospital's own capture. The scorer **must stand alone on video.**
2. **Kinematics / telemetry — enrichment where obtainable.** Vendor-controlled, schema-divergent, and unavailable to third parties on clinical systems by any publicly documented route (§V-1). Treat every kinematics integration as a bilateral partnership deliverable with its own gate. **No roadmap item may be blocked on kinematics.**
3. Simulator telemetry (SimNow, VR sims) — accessible, low-stakes, useful for cold-start calibration.

**Rationale for the inversion (v1 → v2):** a cross-platform scorer cannot depend on a per-vendor, vendor-gated signal. If kinematics were the differentiator, the product could only exist with permission from the parties it competes with. Video-first makes the product buildable and the neutrality claim coherent; kinematics becomes accretive upside on partnered platforms.

**Pipeline:**
1. **Ingestion** — video capture, timestamp alignment, procedure metadata, versioning.
2. **De-identification** — see §8; a named work-stream, not a pipeline step.
3. **Perception** — instrument/tissue/anatomy segmentation, surgical-phase recognition, critical-structure detection (bile duct, ureter, vessels), bleeding detection. Bootstrap from Cholec80/EndoVis.
4. **Safety detection** — Critical View of Safety (Strasberg criteria) achievement, near-miss events, critical-structure proximity. Deterministic where possible.
5. **Skill scoring** — **binary proficiency metrics primary, GEARS domains secondary** (§13 rationale). Calibrated against a multi-rater expert panel.
6. **Attestation output** — vector scorecard, learning curve, credentialing report, audit trail, contestation record.

**Output artifacts:**
- Per-surgeon scorecard (safety events, binary proficiency items, GEARS domains, trends).
- Credentialing report with an **explicit, pre-registered decision rule** (§7.2).
- Regression diff (surgeon progress, or model-version regression for Segment 4).
- Contestation record (§7.3).

### 7.1 Architecture

```
┌─ Ingestion ──────────────────────────────────┐
│ video (OR / simulator) · procedure metadata  │
│ [kinematics: optional, partner-gated]        │
│ timestamp alignment · versioning             │
└──────────────────┬───────────────────────────┘
                   ▼
┌─ De-identification (named work-stream) ──────┐
│ out-of-body segment detection · face/room    │
│ redaction · burned-in overlay (MRN/name/DOB) │
│ removal · audio PHI handling · attestation   │
└──────────────────┬───────────────────────────┘
                   ▼
┌─ Perception (CV models) ─────────────────────┐
│ segmentation · phase · critical-structure    │
│ bleeding · instrument tracking               │
└──────────────────┬───────────────────────────┘
                   ▼
┌─ Scoring engine (layered verifier) ──────────┐
│ [hard]  deterministic safety gates: CVS,     │
│         structure proximity, bleeding        │
│ [soft]  binary proficiency metrics; GEARS    │
│ [human] expert panel on contested cases      │
└──────────────────┬───────────────────────────┘
                   ▼
┌─ Attestation output ─────────────────────────┐
│ vector scorecard · learning curve            │
│ decision rule (§7.2) · contestation (§7.3)   │
│ credentialing report · audit trail           │
└──────────────────────────────────────────────┘
```

- **Score is a vector, never an implicit scalar.** Skill domains and safety events report separately; hard gates stay distinct from soft scores.
- **Data model:** procedure → surgeon → system → episode → aligned video (+kinematics) → annotations → score → decision → contestation.
- **Compliance:** HIPAA/BAA for PHI in video; de-identification pipeline; audit logging; SOC 2 for the hosted layer.

### 7.2 Owning the scalar collapse

Credentialing terminates in a binary decision: grant privileges or don't. Someone **will** collapse the vector. If we don't specify the collapse, each hospital invents an unreviewed one and every resulting dispute lands on our artifact anyway.

Requirements:
- A **pre-registered, versioned decision rule** mapping the vector to `meets-benchmark / does-not-meet / indeterminate`, published before the pilot, changed only by versioned amendment.
- **`indeterminate` is a required output class.** A scorer that cannot abstain will be forced into false confidence exactly where liability concentrates.
- **Named threshold owner.** Document explicitly whether the benchmark threshold is set by us, by the customer, or by a specialty-society standard, and who bears the consequence of it being wrong. Resolve with counsel and reflect in contract (§V-3).
- Hard safety gates never average into soft scores.

### 7.3 Contestability (product requirement, not a nicety)

The subject of an adverse score is a licensed professional with career exposure and a strong incentive to litigate. Ship from v1 of the product:
- Surgeon right-of-access to their own score and the evidence behind it.
- A defined appeals path routing to human expert re-review.
- Surfaced **rater disagreement** — where the expert panel split, the artifact says so.
- Recorded right-of-response attached durably to the artifact.
- Immutable audit trail of score version, model version, decision rule version.

## 8. De-identification (work-stream, own risk row)

v1 treated this as one box. It is a distinct, expensive engineering problem and a gating dependency for every data-rights conversation.

- **Out-of-body segments.** Endoscopic cameras exit the patient mid-procedure and record the room, staff faces, and whiteboards. Detection and redaction of these segments is required, not optional.
- **Burned-in overlays.** Patient name, MRN, DOB, and date are frequently rendered into the video raster by capture systems; they cannot be stripped as metadata.
- **Audio.** Intraoperative audio carries names, identifiers, and clinically sensitive discussion. Default to discard unless a specific use justifies retention.
- **Deliverable:** a de-identification attestation artifact per episode, versioned, auditable, and reviewable by an institution's privacy office. This artifact is a sales asset, not just compliance.

## 9. Legal structure: privilege, discoverability, and regulatory posture

**This is the section v1 omitted entirely, and it may be the largest single determinant of whether the product is buyable.**

The product manufactures durable, attributable, adverse-capable findings about named clinicians. A risk officer's first question is not accuracy — it is *"what happens to this record in litigation?"* Every claim in this section is a **hypothesis requiring health-law counsel** and appears in §V.

**Candidate protective structures to evaluate in Phase 0:**
- **PSQIA / Patient Safety Organization designation.** Federal patient-safety-work-product protections are a plausible fit for retrospective QA use. Applicability, scope, and whether *credentialing* use falls inside or outside the protection must be confirmed — several privilege frameworks treat credentialing differently from quality improvement (§V-3).
- **State peer-review statutes.** Vary substantially by state in scope and in whether external vendors fall within the privileged committee. Requires per-jurisdiction analysis.
- **Contractual and architectural controls** as fallback: customer-held keys, defined retention/destruction schedules, aggregate-only reporting tiers, and a mode where individual attribution is withheld.

**Strategic read:** if a defensible privilege posture is achievable, it is a genuine moat — it is legal and operational work, it compounds per jurisdiction, and it is exactly the kind of unglamorous barrier generalist video-AI vendors skip. If it is *not* achievable for credentialing use, the credentialing wedge narrows toward QA/education and Segment 2 weakens materially. **Phase 0 must answer this before build.**

**Regulatory posture (softened from v1's assertion):**
- The intended use — education, credentialing, retrospective QA — is *positioned* outside the medical-device regime. v1 stated this as fact; it is a **legal position requiring counsel confirmation**, not an established property of the product (§V-2).
- **FDA is likely not the binding Phase 1 constraint.** IRB approval, data-use agreements, state peer-review law, and HIPAA de-identification of video are nearer and harder. Staff accordingly (§14).
- **Non-US regimes must be assessed separately.** EU MDR's software classification and GDPR's treatment of surgical video are materially different questions from the US analysis and must not be assumed to follow it (§V-5, §17).
- **Line discipline:** the moment a customer wants the output to gate a live clinical decision, the intended use shifts and the device regime applies. That is the §11 trigger boundary, not a Phase 1 feature.
- **Assurance positioning:** align output artifacts with CHAI-style model-card / nutrition-label conventions so the platform reads as assurance infrastructure. Matters for Segment 5 later.

## 10. Data strategy & rights

The durable asset is **a cross-platform corpus of robotic surgical video with expert labels and safety annotations, plus the calibrated scorer and rater panel that produce them, held under a legal structure customers can accept.**

- **Acquisition paths:** challenger-platform partnerships (they want independent onboarding/proficiency evidence — Segment 1), credentialing-program partnerships (simulator + proctored cases are the most accessible training data), hospital QA engagements.
- **Labeling flywheel:** expert raters score a subset → calibrate the automated scorer → contested/novel cases route back to raters → gold labels grow → scorer improves. The panel is a cost center *and* part of the moat; grow it from contested cases, not speculatively up front.
- **Annotation economics must be priced before build.** "Annotated procedure" is undefined in v1. Define: annotator qualification, label schema, annotation density (per-frame / per-phase / per-event), redundancy (raters per case), and **fully-loaded cost per annotated procedure.** That unit cost determines whether the flywheel is a moat or a treadmill, and it sets the real meaning of the §11 corpus threshold. **Phase 0 deliverable.**
- **Kinematics rights:** pursue opportunistically via Segment 1 partnerships. High value where obtained, but **explicitly not on the critical path** (§7, §V-1).
- **Segment 4 is a moat leak unless restricted.** Selling labeled ground truth to surgical-AI companies trains a potential replacement scorer. Any data license must carry **field-of-use restrictions barring derivative skill-assessment or safety-scoring products**, or Segment 4 is reclassified from expansion to leakage and dropped.
- **Data rights risk:** surgical video is PHI and institutionally gated. Combined with §9, this is the single biggest execution risk and the reason §11's corpus threshold gates expansion.

## 11. Expansion trigger (explicit, falsifiable)

The wedge expands to **intraoperative AI / regulated validation** only when ALL hold:

1. **Corpus threshold:** **PROVISIONAL — not yet derived.** Placeholder ≥25,000 annotated robotic procedures (annotation defined per §10) spanning **≥2 robot platforms**, with safety-critical-structure labels.
   *Two things are established and one is not. Established: (a) v1's 10,000 equaled the Phase 2 success metric, so it constrained nothing — the trigger must sit strictly beyond the prior phase's target; (b) cross-platform coverage belongs in the trigger because it is the thesis. **Not established: the number itself.** 25,000 is a placeholder chosen only to satisfy (a); no power calculation or cost model backs it, and treating it as settled would be false precision in a kill criterion.*
   ***Phase 0 derivation rule (closes V-7, V-9).** Set the threshold as the maximum of: (i) the corpus size at which a held-out per-Strasberg-criterion sensitivity estimate reaches a pre-specified confidence half-width at the §13 specificity floor, given observed event prevalence — i.e. statistical power, not roundness; (ii) the corpus size at which the §10 fully-loaded annotation unit cost stays inside the funded budget; (iii) any strictly-greater-than-Phase-2 floor implied by (a) above. Record the derivation and the inputs in the trigger before it is treated as binding. Until derived, this condition is **not a valid gate** and no capital decision may cite it.*
2. **Partner pull:** ≥2 signed partnerships with robotic platform companies (Medtronic/CMR/Vicarious/Moon/Distalmotion class).
3. **Regulatory clarity:** FDA PCCP / adaptive-AI guidance sufficiently mature to scope intraoperative validation.
   *Changed from v1: the "OR one device company brings a defined pathway with committed budget" branch is deleted — it was near-identical to condition 4 and reduced four conditions to roughly two and a half.*
4. **Pull, not push:** ≥1 paying customer with a signed commitment for intraoperative validation, at an ACV covering the regulatory build.

The far endgame — certification of **autonomous/assistive surgical robots** — is a monitored hypothesis. Build nothing for it; track (a) autonomous surgical systems entering regulatory review and (b) whether the corpus + safety annotations + neutral position compound to where certification is the natural owner. Until both are visible, it is an option, not a plan.

## 12. Roadmap (wedge first; robotics is gated, not scheduled)

**Phase 0 — Validation (kill-gate, 3–6 months)**

Phase 0 answers four questions, all of which can kill the plan cheaply:

- **Demand gate (strengthened from v1's "≥1 signed LOI"):** ≥3 **paying** design partners, at **distinct institutions**, at or above a stated ACV floor, with **≥1 paying without platform-vendor subsidy.** An LOI is not a signature and not money. v1's single-pilot gate could not distinguish a platform from a services shop — which §18 names as the only question that matters.
- **Legal gate:** counsel opinion on §9 — privilege posture, credentialing-vs-QA treatment, device-regime positioning, and the §V items.
- **Economics gate:** measured fully-loaded cost per annotated procedure (§10); rater panel recruited at viable cost.
- **Data gate:** rights secured to a minimum viable corpus (simulator + proctored video), with de-identification (§8) accepted by at least one institutional privacy office.
- **Written answer to the C-SATS question** (§4).
- **Gate:** proceed only if all five clear.

**Phase 1 — Wedge (6–12 months)**
- Build ingestion + de-identification + CVS detection + proficiency/GEARS scoring + contestability + reporting.
- Calibrate automated scoring against the multi-rater expert panel.
- Ship scorecard + credentialing report to design partners.
- **Accuracy gate:** §13 metrics.
- **Renewal gate (new):** ≥2 of 3 design partners renew at or above initial ACV, **and** ≥1 documented credentialing or QA decision that materially used the output. *The likeliest failure mode is not "the score missed its target" — it is "the score hit its target, nobody changed behavior, nobody renewed." v1 had no gate for this.*

**Phase 2 — Expansion within non-regulated zone (12–24 months)**
- Scale corpus via challenger-platform partnerships; add a second robot platform.
- Hospital QA and learning-curve analytics (Segment 3).
- Segment 4 data products **only under §10 field-of-use restrictions**.

**Phase 3 — Intraoperative AI validation (OPTION, only on §11 trigger)** — not scheduled.

**Phase 4 — Autonomous surgery certification (OPTION, far horizon)** — hypothesis only, no committed work.

## 13. Success metrics (respecified)

v1's gates were under-specified in ways that would have let a weak model pass.

**Skill scoring**
- **Primary endpoint: binary proficiency metrics**, not GEARS. Published comparison work reports GEARS showing good interobserver reliability among experts but degrading with non-experts, and a randomized trial found binary scoring metrics outperformed GEARS on reliability and discrimination ([PMID 37746611](https://pubmed.ncbi.nlm.nih.gov/37746611/), [PMID 25609318](https://pubmed.ncbi.nlm.nih.gov/25609318/)). Building the headline gate on the weaker instrument imports its noise.
- **GEARS as secondary endpoint**, retained for interoperability with existing programs.
- **ICC form must be stated: ICC(2,1), single-rater, absolute agreement.** v1 said "ICC ≥ 0.8" without specifying form; ICC(3,k) with averaged raters can report far higher on identical data. Averaging is prohibited for the headline number.
- **Target is relative, not absolute:** automated-vs-expert ICC(2,1) ≥ **0.90 × (expert-vs-expert ICC(2,1) on the same held-out cases)**, with the expert-vs-expert value reported alongside. The human panel is the ceiling; an absolute target either demands superhuman consistency or silently accepts a weak model when panel agreement is poor.
- **Cohort must be stratified.** The headline ICC is computed on a **within-band cohort** (e.g., practicing attendings only). Novice-vs-expert separation is inflated by between-group variance and is close to trivial; credentialing requires within-band discrimination. Mixed-cohort ICC may be reported as secondary and clearly labeled.

**Safety detection**
- **Replace v1's "CVS AUROC ≥ 0.9."** AUROC on an imbalanced, subjectively-labeled endpoint is prevalence-sensitive and gameable.
- **Per-Strasberg-criterion sensitivity at a fixed, clinically-chosen specificity** (specificity floor set with the clinical lead before evaluation, not after).
- Ground truth = **≥3-rater consensus**, with the raters' own agreement (Fleiss' κ) reported next to the model's score. A model cannot be credited with agreement its labels don't contain.

**Phase-level**
- **Phase 1:** skill and safety gates above; renewal gate (§12); ≥3 paying design partners; ≥500 annotated procedures at a measured, published unit cost.
- **Phase 2:** ≥10k annotated procedures; ≥2 platforms represented; ≥2 challenger-platform partnerships; ≥3 Segment 3/4 customers under field-of-use restriction.
- **Trigger (Phase 3):** the four §11 conditions.

## 14. Business model

- **Per-surgeon / per-program SaaS:** credentialing and learning-curve tracking. Anchor revenue.
- **Platform-partner contracts (Segment 1):** independent proficiency evidence programs; likely the largest early ACV.
- **Per-procedure data products (Segment 4):** only under §10 field-of-use restrictions.
- **Validation services:** later, on trigger — higher ACV, higher overhead.
- **Pricing principle (corrected):** GEARS, OSATS, and GOALS are published, freely available instruments — **the rubric cannot be sold.** What is sold is the **calibrated scorer, the expert panel's attested judgment, the cross-platform comparability, and the legal wrapper (§9).** v1's "charge for the labels, rubrics, and fixtures, not the software" both mispriced the rubric as an asset and contradicted the SaaS anchor above it.

## 15. Risks & kill criteria

| Risk | Severity | Mitigation / kill criterion |
|------|----------|------------------------------|
| **Adverse-finding record is not legally holdable** (privilege/discovery unresolved) | **Critical** | §9 counsel opinion in Phase 0; kill or narrow to de-identified aggregate QA if credentialing use cannot be protected |
| Can't secure data rights (PHI/IRB/institutional gating) | **Critical** | Kill in Phase 0 if minimum viable corpus + privacy-office acceptance can't be secured |
| **Credentialing has mandate but no software budget** | **Critical** | Kill in Phase 0 if the 3-paying-design-partner gate fails, especially the unsubsidized one |
| **Niche floor — buyer count too small** (the C-SATS outcome) | **Critical** | Phase 0 must quantify Segments 1–4 TAM and answer the §4 C-SATS question in writing; if < ~100 credible buyers, it's a services business or a tuck-in, not a platform |
| **An independent (non-manufacturer) body occupies the neutral position first** — specialty society, assurance lab, payer consortium | **High** | *Replaces v1's incumbent row.* Native vendor scoring is now a precondition, not a threat; a credible neutral competitor is the real threat. Monitor SAGES/ACS/Society of Robotic Surgery and CHAI-class programs; partner rather than compete if one emerges |
| Automated score can't reach panel-relative agreement | **High** | Kill wedge if §13 relative-ICC gate fails after calibration |
| **Score is accurate but changes nothing** | **High** | Phase 1 renewal gate (§12) |
| De-identification proves infeasible at acceptable cost | **High** | §8 is a Phase 0 cost line; kill if per-episode cost exceeds pricing headroom |
| Kinematics access never materializes | **Medium** | *Downgraded from v1's critical-path assumption.* Video-first architecture (§7) means this costs upside, not viability |
| Segment 4 trains a replacement | **Medium** | Field-of-use restrictions (§10), or drop the segment |
| Intraop expansion premature (regulatory) | **Medium** | Gated on §11 trigger; do not pre-build |

## 16. Team & build order

1. **Clinical/rater lead** — owns rubric calibration, panel recruitment, score defensibility, and the §13 specificity floor. *Promoted to first: the calibration standard gates everything downstream.*
2. **CV / ML engineer** — perception (segmentation, phase, critical-structure detection). Bootstrap from public datasets.
3. **Data/back-end engineer** — ingestion, de-identification (§8), storage, audit, contestation records.
4. **Enterprise seller into hospital med-exec / credentialing committees and GPOs (new).** This is a slow, relationship-driven committee sale with a procurement cycle no engineer shortens. v1 had no commercial hire, while betting the company on a §12 paying-partner gate.
5. **Health-privacy / peer-review counsel (new, not part-time).** Owns §9 and §V. v1's sole regulatory role was scoped to FDA — likely the *least* binding Phase 1 constraint. Retain a part-time FDA/assurance advisor separately for CHAI alignment and trigger readiness.

**Build order:** de-identification + ingestion → CVS detection (safety gate: most objective, most checkable, most defensible) → binary proficiency + GEARS encoding (calibrate against panel) → decision rule + contestability → scorecard/reporting → corpus scale → opportunistic kinematics enrichment.

## 17. Geography

Evaluate in Phase 0; it may relocate the wedge cheaply.

- **US:** fragmented buyers (per-hospital privileging via med-exec committee), state-by-state peer-review variation, HIPAA.
- **UK/EU:** more centralized procurement (NHS trusts, national programs) — potentially a faster single sale and a better fit for a neutral third-party attestation body. But GDPR treatment of surgical video and EU MDR software classification are materially harsher and **must be analyzed independently rather than inferred from the US position** (§V-5).

A one-week desk assessment in Phase 0 is sufficient to decide whether to lead US or EU.

## V. Open verification items

Claims v1 asserted as fact that are actually **hypotheses with owners and gates.** None may be treated as settled in planning, pricing, or customer-facing material until closed.

| # | Claim to verify | Current basis | Owner | Gate |
|---|---|---|---|---|
| V-1 | Third-party access to kinematics on **clinical** robotic systems | No publicly documented route found. The da Vinci Research Kit is retired first-generation hardware provided under research agreement and marked *not for clinical use* ([arXiv 2104.09869](https://arxiv.org/pdf/2104.09869.pdf), [Intuitive Foundation](https://www.intuitive-foundation.org/dvrk/)). **This is absence of public documentation, not proof of universal unavailability** — bilateral OEM data agreements may exist and are not public. Availability must be confirmed **per vendor**. | ML lead | Phase 0 — direct inquiry to ≥3 platform vendors. Architecture assumes unavailable (§7). |
| V-2 | Wedge intended use sits outside the medical-device regime | Reasoned legal position based on education/credentialing/QA framing. **Not confirmed.** | Counsel | Phase 0 — written opinion before build |
| V-3 | PSQIA/PSO or state peer-review privilege can protect the artifact, **including for credentialing use** | Candidate structures only. Privilege scope varies by state, and credentialing is treated differently from quality improvement in some frameworks. **Unresolved.** | Counsel | Phase 0 — must resolve before the credentialing wedge is committed (§9) |
| V-4 | Credentialing programs hold, or can obtain, budget for assessment software | Credentialing is a mandatory activity; expert consensus supports video-based performance review for privileging ([PMID 33214434](https://pubmed.ncbi.nlm.nih.gov/33214434/)). No source establishes an existing software line item, and vendor subsidy of proctoring is a confound. | Commercial lead | Phase 0 — the unsubsidized paying partner (§12) is the test |
| V-5 | EU MDR / GDPR treatment of the product | Not analyzed. **Must not be inferred from the US position.** | Counsel | Phase 0 desk assessment (§17) |
| V-6 | TAM: credible buyer count across Segments 1–4 | Not established. Directly determines platform-vs-services (§15, §18). | Commercial lead | Phase 0 |
| V-7 | Fully-loaded cost per annotated procedure | Not measured. Sets the real meaning of §11's corpus threshold. | Clinical/rater lead | Phase 0 |
| V-8 | Competitor capability detail (Case Insights metric definitions, scoring rubrics, availability) | Marketing-level description only; specific metrics and rubrics not disclosed publicly. | Commercial lead | Phase 0 — refine via customers who use it |
| V-9 | **§11 corpus threshold (the 25,000 figure)** | **Placeholder, not derived.** Chosen only to sit strictly above the Phase 2 target. No power calculation, prevalence estimate, or cost model supports the specific number. Until derived per the §11 rule, condition 1 is not a valid gate. | Clinical/rater lead + ML lead | Phase 0 — publish derivation (power × unit cost × Phase-2 floor) or revert to the Phase 2 figure |

## 18. Honest framing

The wedge is a real, buildable business, but v2 rests on a narrower and more specific claim than v1: **not that we can score robotic surgery, but that an independent party's score is worth paying for when the robot vendor now issues one for free.**

That is a better thesis than v1's — it survives the incumbent's entry instead of being killed by it, and it explains why the buyer needs a startup rather than a bundled feature. It is also a harder sell, because it depends on neutrality being *valued*, not merely *true*.

Two questions can kill this, both answerable cheaply in Phase 0, neither answerable by building:

1. **Legal holdability (§9, V-3).** If a hospital cannot hold an adverse, attributable, discoverable finding about a named surgeon under a defensible privilege structure, the credentialing wedge collapses toward de-identified aggregate QA and the business shrinks materially.
2. **Niche floor (§15, V-6).** Whether the buyer count across challengers, credentialing programs, hospitals, and surgical-AI companies supports a platform rather than a services shop or a strategic tuck-in. **C-SATS is the prior, and the prior is a tuck-in.** Phase 0 must state in writing what is materially different now.

Everything past Phase 2 — intraoperative AI validation, then autonomous-robotics certification — remains a value hypothesis with an explicit trigger. The plan commits capital to the wedge and the non-regulated expansion, and nothing to robotics certification until §11 fires.
