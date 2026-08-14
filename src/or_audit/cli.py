"""Command-line entry points.

Three commands, each answering a question someone actually asks:

``demo``
    Does the whole chain work? Runs synthetic data end to end and prints the
    credentialing report and the audit summary.
``verify-audit``
    Is this audit log intact? Verifies a chain against an externally pinned
    head hash, which is the only way tail truncation is detectable. Without a
    pin it exits 3 rather than 0: a clean exit code from a check that could not
    see truncation would be read by a script as a full pass, and the pin's
    provenance is the entire security property. ``--allow-unpinned`` is the
    explicit acknowledgement.
``describe-rule``
    What rule will be applied? Prints the decision rule for publication before
    a pilot, as PLAN.md section 7.2 requires.

Written against ``argparse`` rather than a CLI framework: three commands do not
justify a dependency, and this file is small enough to read in one sitting.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from or_audit.audit.trail import AuditTrail
from or_audit.decision.rule import DecisionRule
from or_audit.demo import run_demo
from or_audit.domain.enums import ThresholdOwner
from or_audit.errors import AuditChainError
from or_audit.version import PACKAGE_VERSION

_DISCLAIMER = (
    "NOTE: synthetic data. This repository holds no clinical media, and the "
    "detectors are screening heuristics, not validated classifiers. Nothing "
    "here is evidence of clinical performance (PLAN.md sections 8, 9, V)."
)


def _demo(args: argparse.Namespace) -> int:
    """Run the synthetic end-to-end demo."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(args.workdir) if args.workdir else Path(tmp)
        workdir.mkdir(parents=True, exist_ok=True)
        outcome = run_demo(workdir, episodes=args.episodes)

        print(outcome.report.render())
        print()
        print("AUDIT TRAIL")
        print(f"  entries      {len(outcome.trail)}")
        print(f"  head         {outcome.trail.head_hash}")
        try:
            outcome.trail.verify(
                expected_head=outcome.trail.head_hash, expected_length=len(outcome.trail)
            )
            print("  verification intact (chain + pinned head + pinned length)")
        except AuditChainError as exc:  # pragma: no cover - defensive
            print(f"  verification FAILED: {exc}")
            return 1
        print()
        print("DE-IDENTIFICATION")
        print(f"  segments dropped {outcome.deid_frames_dropped}")
        print(f"  regions masked   {outcome.deid_boxes_masked}")
        if args.audit_log:
            outcome.trail.to_jsonl(Path(args.audit_log))
            print(f"  audit log written to {args.audit_log}")
        print()
        print(_DISCLAIMER)
    return 0


def _verify_audit(args: argparse.Namespace) -> int:
    """Verify an audit log, optionally against a pinned head."""
    path = Path(args.path)
    if not path.is_file():
        print(f"no such audit log: {path}", file=sys.stderr)
        return 2
    try:
        trail = AuditTrail.from_jsonl(path, verify=False)
    except AuditChainError as exc:
        print(f"UNREADABLE: {exc}", file=sys.stderr)
        return 1
    try:
        trail.verify(
            expected_head=args.expected_head,
            expected_length=args.expected_length,
        )
    except AuditChainError as exc:
        print(f"BROKEN: {exc}", file=sys.stderr)
        return 1
    print(f"intact: {len(trail)} entries, head {trail.head_hash}")
    if args.expected_head is None:
        print(
            "  WARNING: no --expected-head supplied. Tail truncation is not "
            "detectable from the chain alone; pin the head externally for this "
            "check to mean anything."
        )
        if not args.allow_unpinned:
            print(
                "  INCOMPLETE: refusing to report a clean result from an "
                "unpinned check. Supply --expected-head, or --allow-unpinned to "
                "acknowledge that truncation was not checked.",
                file=sys.stderr,
            )
            return 3
    return 0


def _describe_rule(args: argparse.Namespace) -> int:
    """Print a decision rule for pre-registration."""
    rule = DecisionRule(
        version=args.version,
        threshold_owner=ThresholdOwner(args.threshold_owner),
        threshold_provenance=args.provenance,
        min_proficiency_fraction=args.min_proficiency,
        min_assessable_items=args.min_items,
    )
    print(rule.describe())
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="or-audit",
        description="Vendor-neutral robotic surgical skill and safety attestation.",
    )
    parser.add_argument("--version", action="version", version=PACKAGE_VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the synthetic end-to-end pipeline")
    demo.add_argument("--episodes", type=int, default=8, help="episodes to assess")
    demo.add_argument("--workdir", help="directory for redacted output (default: temporary)")
    demo.add_argument("--audit-log", help="write the audit trail to this JSONL path")
    demo.set_defaults(func=_demo)

    verify = sub.add_parser("verify-audit", help="verify an audit log")
    verify.add_argument("path", help="JSONL audit log")
    verify.add_argument(
        "--expected-head", help="externally pinned head hash; required to detect truncation"
    )
    verify.add_argument("--expected-length", type=int, help="externally pinned entry count")
    verify.add_argument(
        "--allow-unpinned",
        action="store_true",
        help=(
            "accept a verification with no pinned head. Exits 3 without this, "
            "because an unpinned check cannot detect truncation and a clean "
            "exit code would overstate what was verified."
        ),
    )
    verify.set_defaults(func=_verify_audit)

    describe = sub.add_parser("describe-rule", help="print a decision rule for publication")
    describe.add_argument("--version", default="1")
    describe.add_argument(
        "--threshold-owner",
        default=ThresholdOwner.CUSTOMER.value,
        choices=[owner.value for owner in ThresholdOwner],
    )
    describe.add_argument(
        "--provenance", default="Credentialing committee minute", help="threshold provenance"
    )
    describe.add_argument("--min-proficiency", type=float, default=0.85)
    describe.add_argument("--min-items", type=int, default=5)
    describe.set_defaults(func=_describe_rule)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
