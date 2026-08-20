# Security Policy for SurgEval

## Reporting a vulnerability

Please report security issues privately — do **not** open a public issue. Email
the SeldingerMed maintainers, or open a GitHub Security Advisory on this
repository (private). Provide:

- the affected version(s) and where you found the issue,
- a minimal reproduction,
- whether it affects task/agent evaluation integrity, registry package pulls,
  or the audit trail.

We acknowledge reports within 5 business days and aim for a fix + advisory
within 30 for critical issues.

## Supported versions

Security fixes land on the latest release and are backported on request for the
current and previous minor release.

## Known trust boundaries

- **Evaluation integrity.** Replay is only trusted for vectors reconstructed
  through the bundled task verifier against a stored trace. A mismatched vector,
  package digest, or projection head is refused. Do not weaken these checks.
- **Registry pulls.** Packages from `SeldingerMed/seldinger-tasks` are pinned by
  git ref **and** tree content digest at materialize time. Keep those pins
  input-only and never bypass `materialize_entry` for untrusted sources.
- **Release artifacts.** Publisher provenance is verified out-of-band at the
  asset layer via Sigstore: GitHub build attestations (`gh attestation verify`)
  and PyPI PEP 740 attestations. There is no in-band signing key in this
  repository; see `docs/OSS_GOVERNANCE.md`.
- **Commercial supply chain.** Runtime dependencies must stay within the
  license allowlist (`scripts/check_license_allowlist.py`). Do not add copyleft
  or unreviewed dependencies.

## Responsible disclosure

We will not pursue legal action against researchers who follow coordinated
disclosure: report privately, give us a reasonable window, and do not exfiltrate
data or exceed a narrowly scoped proof of concept.
