"""P4 registry, portable artifact, and scorecard contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from or_audit.cli import main
from or_audit.errors import TaskContractError
from or_audit.eval.integrity import tree_digest
from or_audit.eval.registry import RegistryEntry, materialize_entry

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DATASET = ROOT / "docs" / "examples" / "datasets" / "video-nextstep-v0"
VIDEO_AGENT = ROOT / "docs" / "examples" / "agents" / "example-video-predictor"


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "format_version": "1",
                "datasets": [
                    {
                        "kind": "dataset",
                        "id": "seldingermed/video-nextstep",
                        "version": "0",
                        "repository": str(ROOT),
                        "ref": "local-test",
                        "path": str(VIDEO_DATASET.relative_to(ROOT)),
                        "digest": tree_digest(VIDEO_DATASET),
                    }
                ],
                "agents": [
                    {
                        "kind": "agent",
                        "id": "example/video-predictor",
                        "version": "0",
                        "repository": str(ROOT),
                        "ref": "local-test",
                        "path": str(VIDEO_AGENT.relative_to(ROOT)),
                        "digest": tree_digest(VIDEO_AGENT),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_registry_ids_cannot_escape_pull_root() -> None:
    with pytest.raises(ValidationError):
        RegistryEntry(
            kind="agent",
            id="../../outside",
            version="0",
            repository="/tmp/repo",
            ref="abc",
            path="agent",
            digest="abc",
        )


def test_registry_package_digest_is_enforced(tmp_path: Path) -> None:
    package = tmp_path / "repo" / "package"
    package.mkdir(parents=True)
    (package / "payload").write_text("first", encoding="utf-8")
    entry = RegistryEntry(
        kind="agent",
        id="example/agent",
        version="0",
        repository=str(tmp_path / "repo"),
        ref="local",
        path="package",
        digest=tree_digest(package),
    )
    assert materialize_entry(entry) == package
    (package / "payload").write_text("changed", encoding="utf-8")
    with pytest.raises(TaskContractError, match="digest mismatch"):
        materialize_entry(entry)


def test_registry_list_pull_and_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registry = _registry(tmp_path)
    assert main(["datasets", "list", "--registry", str(registry)]) == 0
    assert "seldingermed/video-nextstep@0" in capsys.readouterr().out
    pull = tmp_path / "pull"
    assert (
        main(
            [
                "agents",
                "pull",
                "example/video-predictor@0",
                "--registry",
                str(registry),
                "--out",
                str(pull),
            ]
        )
        == 0
    )
    assert (pull / "example" / "video-predictor" / "0" / "agent.toml").is_file()

    out = tmp_path / "run"
    assert (
        main(
            [
                "run",
                "-d",
                "seldingermed/video-nextstep@0",
                "-a",
                "example/video-predictor@0",
                "--registry",
                str(registry),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    job = out / "video-nextstep"
    config = json.loads((job / "config.json").read_text(encoding="utf-8"))
    assert config["task_dir"] == "bundle/task"
    assert config["agent_dir"] == "bundle/agent"
    assert main(["replay", str(job)]) == 0
    config["task_dir"] = "../outside"
    (job / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert main(["replay", str(job)]) == 1
    assert "escapes job directory" in capsys.readouterr().err


def test_static_leaderboard_keeps_vector_columns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _registry(tmp_path)
    run = tmp_path / "run"
    assert (
        main(
            [
                "run",
                "-d",
                "seldingermed/video-nextstep@0",
                "-a",
                "example/video-predictor@0",
                "--registry",
                str(registry),
                "--out",
                str(run),
            ]
        )
        == 0
    )
    site = tmp_path / "site"
    assert main(["leaderboard", str(run), "--out", str(site)]) == 0
    data = json.loads((site / "leaderboard.json").read_text(encoding="utf-8"))
    row = data["rows"][0]
    assert row["headline"] == "next_step_correct"
    assert row["headline_unassessable"] == 1
    assert "abstained" in row["metrics"]
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "Gate failures" in html
    assert "no cross-task overall score" in html
    capsys.readouterr()
    config_path = run / "video-nextstep" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for task_dir, message in (
        (str(run.resolve()), "must be relative"),
        ("../outside", "escapes job directory"),
    ):
        config["task_dir"] = task_dir
        config_path.write_text(json.dumps(config), encoding="utf-8")
        assert main(["leaderboard", str(run), "--out", str(tmp_path / "refused")]) == 1
        assert message in capsys.readouterr().err
    config["task_dir"] = "bundle/task"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    trajectory = run / "video-nextstep" / "trial-video-nextstep-0" / "trajectory.json"
    evidence = json.loads(trajectory.read_text(encoding="utf-8"))
    evidence[0]["prediction"]["next_step"] = "tampered"
    trajectory.write_text(json.dumps(evidence), encoding="utf-8")
    assert main(["leaderboard", str(run), "--out", str(tmp_path / "refused")]) == 1
    assert "reconstitutes a different vector" in capsys.readouterr().err
