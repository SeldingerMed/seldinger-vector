# Dataset Licensing and Data Usage

SurgEval evaluates models against versioned task packages. Each task package may reference input data (video clips, sensor traces, simulator configurations) and oracle labels. This document defines the licensing and data-usage framework that governs how datasets enter the platform.

## PHI classification

Every task and taskset declares a `phi_class` field. The platform enforces this classification at load time and at run time.

| Class | Description | Public pool | Private pool |
|---|---|---|---|
| `public` | Synthetic or publicly released data with no patient identifiers. | Allowed | Allowed |
| `procedural` | Simulated or phantom data that contains no real patient data but may include institution-specific operational details. | Allowed | Allowed |
| `deidentified_clinical` | Clinical data that has been deidentified under HIPAA Safe Harbor or Expert Determination. | **Refused** | Allowed under BAA |
| `prohibited` | Data that cannot be processed on any shared infrastructure (e.g., identifiable PHI, restricted-use research data). | **Refused** | **Refused** |

The loader rejects a task whose `phi_class` is `prohibited` on any pool. The public pool additionally refuses `deidentified_clinical` tasks. A private pool may accept `deidentified_clinical` tasks only when a Business Associate Agreement (BAA) is in place between the hosting operator and the data-providing institution.

## Data usage agreements (DUA)

### For dataset publishers

A dataset publisher is the `org/` entity that authors and pins a task package. The publisher attests:

1. **Provenance**: The publisher holds or has secured the rights to distribute the input data and labels under the declared `phi_class`.
2. **Consent**: If the data derives from human subjects, the publisher confirms IRB approval and informed consent cover the evaluation use case.
3. **Deidentification**: If `phi_class` is `deidentified_clinical`, the publisher documents the deidentification method (Safe Harbor or Expert Determination) and the date of the determination.
4. **Licensing**: The publisher states the data license in the task's `instruction.md`. Acceptable licenses include CC-BY, CC-BY-SA, CC-BY-NC, Apache-2.0, MIT, or a named institutional DUA. Proprietary or restricted licenses must reference a signed agreement.
5. **Claim boundary**: The publisher defines the scorecard claim footer, which bounds what a result on this dataset can and cannot assert about clinical performance.

### For evaluation consumers

An evaluation consumer is any party that runs a SurgEval job against a published dataset. The consumer agrees:

1. **No redistribution**: The consumer does not redistribute input data or labels beyond the scope of the original license.
2. **No re-identification**: The consumer does not attempt to re-identify individuals in deidentified datasets.
3. **Attribution**: Published results cite the dataset by its `org/name@version` reference and digest.
4. **Scope**: The consumer uses the evaluation results only for the purpose stated in the dataset's claim boundary.

## Dataset packaging

A dataset (v0.2 term) or taskset (v0.3 term) is a directory containing:

```
org/name/version/
  taskset.toml          # or dataset.toml (v0.2 compatibility)
  tasks/
    task-a/
      instruction.md
      task.toml
      inputs.jsonl
      labels.jsonl
      verifier.py
    task-b/
      ...
```

Each task pins its inputs and labels by content digest. The taskset pins each task by tree digest. The registry pins each taskset by git ref and top-level digest. This three-level pinning ensures that a published evaluation result is reproducible: the same taskset, task, and data will produce the same vector for the same agent.

## Synthetic data and simulation provenance

Tasks that use simulation backends (SOFA, Warp, Gymnasium) declare the engine in `WorldSpec`. When `synthetic_stub = true`, the scorecard is stamped with a synthetic banner and `export_rl` refuses to produce RL training data from the run. This prevents synthetic-stub results from being used as RL training signals.

## Current dataset inventory

| Dataset | PHI class | License | Status |
|---|---|---|---|
| `seldingermed/video-nextstep` | `public` | Apache-2.0 (task code) | Seed dataset, runnable |
| `seldingermed/lumen-nav` | `public` | Apache-2.0 (task code) | Seed dataset, runnable via Lumen |
| `seldingermed/angiostress` | `public` | Apache-2.0 (task code) | Seed dataset, claim-bounded |

Third-party dataset publishers are responsible for their own licensing. The platform does not host or distribute third-party data; it evaluates agents against data the publisher has made available under their own terms.

## Open items

- [ ] Reserve formal DUA template for institutional partners.
- [ ] Define audit trail for `deidentified_clinical` access on private pools.
- [ ] Document the claim-footer schema for clinical datasets.
