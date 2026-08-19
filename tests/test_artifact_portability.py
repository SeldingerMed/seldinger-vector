"""Artifact portability: job outputs must be relocatable and free of host paths."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from or_audit.cli import main
from or_audit.eval.integrity import tree_digest
from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.runner import run_job

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DATASET = ROOT / "docs" / "examples" / "datasets" / "video-nextstep-v0"
VIDEO_AGENT = ROOT / "docs" / "examples" / "agents" / "example-video-predictor"
CHOLEC_TASK = ROOT / "docs" / "examples" / "tasks" / "laparoscopic-cholec-cvs"
CVS_AGENT = ROOT / "docs" / "examples" / "agents" / "example-cvs-detector"

# Patterns that indicate an absolute host path leaked into an artifact.
_HOST_PATH_RE = re.compile(
    r"(?:"
    r"/Users/[^/\s\"']+"  # macOS home
    r"|/home/[^/\s\"']+"  # Linux home
    r"|C:\\\\[^\\s\"']+"  # Windows drive
    r"|/private/var/[^/\s\"']+"  # macOS tmp
    r")"
)


def _scan_for_host_paths(directory: Path) -> list[str]:
    """Scan all text-like files in *directory* for embedded absolute host paths."""
    violations: list[str] = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in (".pkl", ".png", ".jpg", ".jpeg", ".bin"):
            continue  # skip binary files
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in _HOST_PATH_RE.finditer(text):
            violations.append(f"{path.relative_to(directory)}: found host path '{match.group(0)}'")
    return violations


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


def _run_cvs_job(tmp_path: Path, subdir: str) -> Path:
    """Run the CVS task with the reference agent and return the job directory."""
    task = load_task(CHOLEC_TASK)
    agent = load_agent(CVS_AGENT)
    out = tmp_path / subdir
    run_job(task=task, task_dir=CHOLEC_TASK, agent=agent, agent_dir=CVS_AGENT, out=out, n=1)
    return out


# ---------------------------------------------------------------------------
# Relocation tests — physically move the job dir and verify replay still works
# ---------------------------------------------------------------------------


def test_job_output_relocatable_cli(tmp_path: Path) -> None:
    """A job directory must replay after being moved to a different parent."""
    registry = _registry(tmp_path)
    out = tmp_path / "original_run"
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
    assert (job / "config.json").is_file()

    # Move the entire job to a different parent (don't pre-create the leaf
    # so shutil.move creates it rather than nesting inside).
    new_parent = tmp_path / "relocated" / "deep" / "nested"
    new_parent.mkdir(parents=True, exist_ok=True)
    relocated = new_parent / "job"
    shutil.move(str(job), str(relocated))

    assert (relocated / "config.json").is_file()
    assert main(["replay", str(relocated)]) == 0


def test_job_output_relocatable_cvs(tmp_path: Path) -> None:
    """A run_job-generated CVS job must replay after relocation."""
    out = _run_cvs_job(tmp_path, "cvs_original")
    assert (out / "config.json").is_file()

    new_parent = tmp_path / "cvs_relocated" / "elsewhere"
    new_parent.mkdir(parents=True, exist_ok=True)
    relocated = new_parent / "job"
    shutil.move(str(out), str(relocated))

    assert (relocated / "config.json").is_file()
    assert main(["replay", str(relocated)]) == 0


# ---------------------------------------------------------------------------
# Host-path leakage tests — no absolute host paths in any artifact file
# ---------------------------------------------------------------------------


def test_no_host_paths_in_cli_job(tmp_path: Path) -> None:
    """CLI-generated job artifacts must not embed absolute host paths."""
    registry = _registry(tmp_path)
    out = tmp_path / "hostpath_cli"
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
    violations = _scan_for_host_paths(out)
    assert not violations, f"Found host paths in CLI job artifacts:\n{chr(10).join(violations)}"


def test_no_host_paths_in_cvs_job(tmp_path: Path) -> None:
    """The CVS example agent job must not embed absolute host paths."""
    out = _run_cvs_job(tmp_path, "hostpath_cvs")
    violations = _scan_for_host_paths(out)
    assert not violations, f"Found host paths in CVS job artifacts:\n{chr(10).join(violations)}"


def test_no_host_paths_in_sdk_job(tmp_path: Path) -> None:
    """SDK-generated job artifacts must not embed absolute host paths."""
    import surgeval as se

    class PortableModel:
        def predict(self, item: dict[str, Any]) -> dict[str, Any]:
            del item
            return {"cvs_achieved": False, "critical_structure": "common_bile_duct"}

    out = tmp_path / "hostpath_sdk"
    se.evaluate(PortableModel(), CHOLEC_TASK, out=out, n=1)
    violations = _scan_for_host_paths(out)
    assert not violations, f"Found host paths in SDK job artifacts:\n{chr(10).join(violations)}"


# ---------------------------------------------------------------------------
# Bundle integrity — config paths are relative and digest-verified
# ---------------------------------------------------------------------------


def test_config_paths_are_relative(tmp_path: Path) -> None:
    """config.json must use relative paths for task_dir and agent_dir."""
    registry = _registry(tmp_path)
    out = tmp_path / "relpaths"
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
    assert not Path(config["task_dir"]).is_absolute(), "task_dir must be relative"
    if config.get("agent_dir"):
        assert not Path(config["agent_dir"]).is_absolute(), "agent_dir must be relative"


def test_bundle_manifest_uses_relative_paths(tmp_path: Path) -> None:
    """bundle.json must use relative paths."""
    registry = _registry(tmp_path)
    out = tmp_path / "manifest"
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
    manifest = json.loads((job / "bundle.json").read_text(encoding="utf-8"))
    assert manifest["task"]["path"] == "bundle/task"
    assert manifest["agent"]["path"] == "bundle/agent"
