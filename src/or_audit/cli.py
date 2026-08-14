"""Command-line entry points.

``demo`` / ``verify-audit`` / ``describe-rule``
    The original credentialing-mode tools. Still work; they are not the wedge.
``tasks validate`` / ``tasks describe`` / ``datasets validate`` /
``agents validate`` / ``bind``
    BUILD.md P0: load the Harbor-shaped eval contract without talking to Lumen.

Written against ``argparse`` rather than a CLI framework.
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
from or_audit.errors import AuditChainError, TaskContractError
from or_audit.eval.bind import assert_bind
from or_audit.eval.loader import load_agent, load_dataset, load_task
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


def _tasks_validate(args: argparse.Namespace) -> int:
    """Load a task directory and exit 0 only if the contract holds."""
    try:
        task = load_task(Path(args.path))
    except TaskContractError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    runnable = "runnable" if task.environment.world_pin else "valid (unpinned, not runnable)"
    print(f"valid: {task.id}@{task.task_version} {runnable}")
    return 0


def _tasks_describe(args: argparse.Namespace) -> int:
    """Print a task the way Harbor would print a task.toml."""
    try:
        task = load_task(Path(args.path))
    except TaskContractError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(task.describe())
    print()
    print(task.instruction)
    return 0


def _datasets_validate(args: argparse.Namespace) -> int:
    """Load a dataset and every task it names."""
    try:
        dataset = load_dataset(Path(args.path))
    except TaskContractError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(
        f"valid: {dataset.id}@{dataset.dataset_version} "
        f"{len(dataset.tasks)} task(s), headline {dataset.headline}"
    )
    return 0


def _agents_validate(args: argparse.Namespace) -> int:
    """Load an org/name agent package."""
    try:
        agent = load_agent(Path(args.path))
    except TaskContractError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"valid: {agent.id}@{agent.agent_version} port={agent.port.value}")
    return 0


def _bind(args: argparse.Namespace) -> int:
    """Refuse a (task, agent) pair whose ports do not match."""
    try:
        task = load_task(Path(args.task))
        agent = load_agent(Path(args.agent))
        assert_bind(task, agent)
    except TaskContractError as exc:
        print(f"INCOMPATIBLE: {exc}", file=sys.stderr)
        return 1
    print(f"bind: {agent.id} -> {task.id} port={task.port.id.value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="or-audit",
        description="Eval harness for procedural medical AI (Harbor analog). "
        "Also still runs the synthetic credentialing demo.",
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

    tasks = sub.add_parser("tasks", help="validate or describe a Harbor-shaped eval task")
    tasks_sub = tasks.add_subparsers(dest="tasks_command", required=True)
    tasks_validate = tasks_sub.add_parser("validate", help="load and check a task directory")
    tasks_validate.add_argument("path", help="task directory or task.toml")
    tasks_validate.set_defaults(func=_tasks_validate)
    tasks_describe = tasks_sub.add_parser("describe", help="print a task's contract")
    tasks_describe.add_argument("path", help="task directory or task.toml")
    tasks_describe.set_defaults(func=_tasks_describe)

    datasets = sub.add_parser("datasets", help="validate a dataset of eval tasks")
    datasets_sub = datasets.add_subparsers(dest="datasets_command", required=True)
    datasets_validate = datasets_sub.add_parser("validate", help="load a dataset and its tasks")
    datasets_validate.add_argument("path", help="dataset directory or dataset.toml")
    datasets_validate.set_defaults(func=_datasets_validate)

    agents = sub.add_parser("agents", help="validate an org/name agent package")
    agents_sub = agents.add_subparsers(dest="agents_command", required=True)
    agents_validate = agents_sub.add_parser("validate", help="load and check an agent directory")
    agents_validate.add_argument("path", help="agent directory or agent.toml")
    agents_validate.set_defaults(func=_agents_validate)

    bind = sub.add_parser(
        "bind",
        help="check that an agent implements the port a task requires",
    )
    bind.add_argument("task", help="task directory or task.toml")
    bind.add_argument("agent", help="agent directory or agent.toml")
    bind.set_defaults(func=_bind)

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
