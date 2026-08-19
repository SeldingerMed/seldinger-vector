"""End-to-end test for the laparoscopic CVS task with the reference CVS detector agent."""

from __future__ import annotations

import json
from pathlib import Path

from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.runner import run_job

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = REPO_ROOT / "docs/examples/tasks/laparoscopic-cholec-cvs"
AGENT_DIR = REPO_ROOT / "docs/examples/agents/example-cvs-detector"


def test_cvs_bind_succeeds_without_wildcard() -> None:
    """The reference agent must bind to the CVS task without a schema wildcard."""
    from or_audit.eval.bind import assert_bind

    task = load_task(TASK_DIR)
    agent = load_agent(AGENT_DIR)
    cap = next(c for c in agent.capabilities if c.interface == task.interface.id)
    assert not cap.schema_wildcard, "reference agent must not use schema_wildcard"
    assert_bind(task, agent)


def test_cvs_run_and_replay(tmp_path: Path) -> None:
    """Run the CVS task end-to-end and verify replay matches."""
    task = load_task(TASK_DIR)
    agent = load_agent(AGENT_DIR)
    out = tmp_path / "cvs-run"
    result = run_job(task=task, task_dir=TASK_DIR, agent=agent, agent_dir=AGENT_DIR, out=out, n=5)
    assert result.n == 5
    assert len(result.trials) == 5
    for trial in result.trials:
        assert len(trial.vector.gates) >= 1
        assert trial.vector.gates[0].id == "critical_structure_misid"
        assert len(trial.vector.metrics) >= 3
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["modality_adapter"]["modality"] == "video-laparoscopic"
    assert config["modality_adapter"]["adapter"] == "VideoAdapter"
    assert (out / "scorecard.json").exists()
    assert (out / "scorecard.md").exists()
    assert (out / "scorecard.html").exists()


def test_cvs_adapter_on_execution_path(tmp_path: Path) -> None:
    """The agent must receive the adapter's preprocessed observation, not the raw item."""
    task = load_task(TASK_DIR)
    agent = load_agent(AGENT_DIR)
    out = tmp_path / "cvs-adapter-check"
    result = run_job(task=task, task_dir=TASK_DIR, agent=agent, agent_dir=AGENT_DIR, out=out, n=1)
    trajectory = result.trials[0].trajectory
    first_step = trajectory[0]
    obs = first_step["obs"]
    assert isinstance(obs, dict)
    assert "clip_id" in obs, "agent must receive adapter output with clip_id, not raw item with id"
    assert "modality" in obs
    assert obs["modality"] == "video-laparoscopic"
