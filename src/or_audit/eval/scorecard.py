"""Deterministic human-readable scorecards for vector-valued eval jobs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from or_audit.eval.job import JobResult


def scorecard_data(result: JobResult) -> dict[str, Any]:
    """Aggregate each gate and metric independently; never create a composite score."""
    gate_ids = [gate.id for gate in result.trials[0].vector.gates]
    metric_ids = [metric.id for metric in result.trials[0].vector.metrics]
    gates = []
    for gate_id in gate_ids:
        statuses = [trial.vector.gate(gate_id).status.value for trial in result.trials]  # type: ignore[union-attr]
        gates.append(
            {
                "id": gate_id,
                "pass": statuses.count("pass"),
                "fail": statuses.count("fail"),
                "not_assessable": statuses.count("not_assessable"),
                "not_applicable": statuses.count("not_applicable"),
            }
        )
    metrics = []
    for metric_id in metric_ids:
        values = [trial.vector.metric(metric_id).value for trial in result.trials]  # type: ignore[union-attr]
        assessed = [value for value in values if value is not None]
        row: dict[str, Any] = {
            "id": metric_id,
            "headline": metric_id == result.headline,
            "assessed": len(assessed),
            "unassessable": len(values) - len(assessed),
        }
        if all(isinstance(value, bool) for value in assessed):
            row.update(
                {
                    "kind": "boolean",
                    "true": assessed.count(True),
                    "false": assessed.count(False),
                    "rate": assessed.count(True) / len(assessed) if assessed else None,
                }
            )
        else:
            numeric = [float(value) for value in assessed if not isinstance(value, bool)]
            row.update(
                {
                    "kind": "numeric",
                    "mean": fmean(numeric) if numeric else None,
                    "min": min(numeric) if numeric else None,
                    "max": max(numeric) if numeric else None,
                }
            )
        metrics.append(row)
    return {
        "task_id": result.task_id,
        "task_version": result.task_version,
        "task_digest": result.task_digest,
        "agent_identity": result.agent_identity,
        "agent_digest": result.agent_digest,
        "world_pin": result.world_pin,
        "n": result.n,
        "headline": result.headline,
        "gates": gates,
        "metrics": metrics,
        "claim_footer": result.claim_footer,
        "head": result.head,
    }


def render_markdown(result: JobResult) -> str:
    data = scorecard_data(result)
    lines = [
        f"# OR-Audit scorecard: {data['task_id']}",
        "",
        f"- Agent: `{data['agent_identity']}`",
        f"- Trials: `{data['n']}`",
        f"- World pin: `{data['world_pin'] or 'none'}`",
        f"- Task digest: `{data['task_digest']}`",
        f"- Agent digest: `{data['agent_digest']}`",
        f"- Artifact head: `{data['head']}`",
        "",
        "## Safety gates",
        "",
        "| Gate | Pass | Fail | Not assessable | Not applicable |",
        "|---|---:|---:|---:|---:|",
    ]
    for gate in data["gates"]:
        lines.append(
            f"| {gate['id']} | {gate['pass']} | {gate['fail']} | "
            f"{gate['not_assessable']} | {gate['not_applicable']} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Headline | Result | Assessed | Unassessable |",
            "|---|:---:|---:|---:|---:|",
        ]
    )
    for metric in data["metrics"]:
        if metric["kind"] == "boolean":
            value = "n/a" if metric["rate"] is None else f"{metric['rate']:.6f}"
        else:
            value = "n/a" if metric["mean"] is None else f"{metric['mean']:.6f}"
        lines.append(
            f"| {metric['id']} | {'yes' if metric['headline'] else 'no'} | {value} | "
            f"{metric['assessed']} | {metric['unassessable']} |"
        )
    if data["claim_footer"]:
        lines.extend(["", "## Claim boundary", "", data["claim_footer"]])
    lines.extend(
        [
            "",
            "> Safety gates and metrics are reported separately. "
            "This scorecard has no composite score.",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(result: JobResult) -> str:
    markdown = render_markdown(result)
    payload = html.escape(json.dumps(scorecard_data(result), indent=2))
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>OR-Audit scorecard</title><style>body{font:15px system-ui;max-width:960px;"
        "margin:40px auto;padding:0 20px;color:#172033}pre{white-space:pre-wrap;background:#f4f6f8;"
        "padding:20px;border-radius:8px}details{margin-top:24px}</style>"
        f"<body><pre>{html.escape(markdown)}</pre><details>"
        "<summary>Machine-readable vector</summary>"
        f"<pre>{payload}</pre></details></body></html>\n"
    )


def write_scorecards(out: Path, result: JobResult) -> None:
    """Write stable Markdown, HTML, and JSON scorecard surfaces."""
    (out / "scorecard.md").write_text(render_markdown(result), encoding="utf-8")
    (out / "scorecard.html").write_text(render_html(result), encoding="utf-8")
    (out / "scorecard.json").write_text(
        json.dumps(scorecard_data(result), indent=2) + "\n", encoding="utf-8"
    )
