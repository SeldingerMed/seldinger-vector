"""P0 eval contract: Harbor-shaped tasks that cannot collapse to a scalar."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from or_audit.cli import main
from or_audit.domain.enums import GateStatus
from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.loader import load_dataset, load_task
from or_audit.eval.task import ProjectionSpec, TaskSpec
from or_audit.eval.vector import TrialVector, project, vector_from_lumen_info

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_TASK = ROOT / "docs" / "examples" / "tasks" / "lumen-nav-safe"
EXAMPLE_DATASET = ROOT / "docs" / "examples" / "datasets" / "lumen-nav-v0"


def _copy_task(tmp_path: Path) -> Path:
    dest = tmp_path / "task"
    shutil.copytree(EXAMPLE_TASK, dest)
    return dest


def _patch_toml(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {path}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


class TestExampleTaskLoads:
    def test_lumen_nav_safe_is_valid(self):
        task = load_task(EXAMPLE_TASK)
        assert task.id == "lumen-nav-safe"
        assert task.verifier.headline == "safe_success"
        assert task.projection is not None
        assert task.projection.id.value == "gated_reach_v0"
        assert "safe success" in task.instruction.lower()

    def test_task_toml_path_also_loads(self):
        assert load_task(EXAMPLE_TASK / "task.toml").id == "lumen-nav-safe"

    def test_missing_task_is_rejected(self, tmp_path):
        with pytest.raises(TaskContractError, match="missing"):
            load_task(tmp_path / "task.toml")

    def test_unpinned_task_is_valid_but_not_runnable(self):
        task = load_task(EXAMPLE_TASK)
        with pytest.raises(TaskContractError, match="world_pin"):
            task.assert_runnable()

    def test_pinned_task_is_runnable(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'world_pin = ""', 'world_pin = "abc123def"')
        load_task(dest).assert_runnable()

    def test_describe_names_the_headline_and_refuses_human_det(self):
        text = load_task(EXAMPLE_TASK).describe()
        assert "headline   safe_success" in text
        assert "human det. refused" in text
        assert "unpinned" in text


class TestContractRefusals:
    def test_raw_success_cannot_headline_when_safe_success_exists(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'headline = "safe_success"', 'headline = "raw_success"')
        with pytest.raises(TaskContractError, match="CathSim"):
            load_task(dest)

    def test_safety_critical_task_without_gates_is_rejected(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(
            dest / "task.toml",
            '[[verifier.gates]]\nid = "wall_penetration"\nsource = "lumen.info.max_pen"\n'
            'fail_when = "max_pen > safety_max_pen"\nmaps_to = "unsafe"\n',
            "",
        )
        with pytest.raises(TaskContractError, match="safety_critical"):
            load_task(dest)

    def test_human_determination_is_refused(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(
            dest / "task.toml",
            "emit_human_determination = false",
            "emit_human_determination = true",
        )
        with pytest.raises(TaskContractError, match="human determination"):
            load_task(dest)

    def test_human_subject_is_refused(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'kind = "policy"', 'kind = "human"')
        with pytest.raises(TaskContractError, match=r"subject\.kind=human"):
            load_task(dest)

    def test_procedural_phi_cannot_attest(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'level = "none"', 'level = "attested"')
        with pytest.raises(TaskContractError, match="attestation"):
            load_task(dest)

    def test_prohibited_phi_cannot_load(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'class = "procedural"', 'class = "prohibited"')
        with pytest.raises(TaskContractError, match="prohibited"):
            load_task(dest)

    def test_lumen_gym_requires_gym_id(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'gym_id = "Lumen/NavTreeBranch-v0"', 'gym_id = ""')
        with pytest.raises(TaskContractError, match="gym_id"):
            load_task(dest)

    def test_missing_instruction_is_rejected(self, tmp_path):
        dest = _copy_task(tmp_path)
        (dest / "instruction.md").unlink()
        with pytest.raises(TaskContractError, match=r"instruction\.md"):
            load_task(dest)


class TestDataset:
    def test_example_dataset_loads_the_task(self):
        dataset = load_dataset(EXAMPLE_DATASET)
        assert dataset.id == "lumen-nav-v0"
        assert dataset.headline == "safe_success"
        assert len(dataset.tasks) == 1
        assert dataset.tasks[0].id == "lumen-nav-safe"

    def test_headline_mismatch_is_rejected(self, tmp_path):
        dest = tmp_path / "ds"
        dest.mkdir()
        (dest / "dataset.toml").write_text(
            "\n".join(
                [
                    'format_version = "1"',
                    'id = "bad-headline"',
                    'dataset_version = "0"',
                    'headline = "max_pen"',
                    'phi_class = "procedural"',
                    "tasks = [",
                    f'    "{EXAMPLE_TASK.as_posix()}",',
                    "]",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(TaskContractError, match="headlines"):
            load_dataset(dest)


class TestTrialVector:
    def _vector(self, info: dict[str, object]) -> TrialVector:
        task = load_task(EXAMPLE_TASK)
        return vector_from_lumen_info(
            task=task,
            agent_identity="random@0",
            seed=0,
            info=info,
            safety_max_pen=0.3,
        )

    def test_safe_success_is_the_headline(self):
        vector = self._vector({"success": True, "safe_success": True, "max_pen": 0.0})
        assert vector.headline.id == "safe_success"
        assert vector.headline.value is True
        assert not vector.any_gate_failed

    def test_raw_reach_with_wall_injury_fails_the_gate(self):
        vector = self._vector(
            {"success": True, "safe_success": False, "unsafe": True, "max_pen": 0.9}
        )
        raw = vector.metric("raw_success")
        assert raw is not None
        assert raw.value is True
        assert vector.headline.value is False
        assert vector.any_gate_failed

    def test_diverged_fails_the_gate(self):
        vector = self._vector({"success": False, "diverged": True, "max_pen": 0.0})
        assert vector.any_gate_failed
        diverged = vector.metric("diverged")
        assert diverged is not None
        assert diverged.value is True

    def test_float_raises(self):
        vector = self._vector({"success": True, "safe_success": True, "max_pen": 0.0})
        with pytest.raises(ScoreContractError, match="no scalar"):
            float(vector)

    def test_int_raises(self):
        vector = self._vector({"success": True, "safe_success": True, "max_pen": 0.0})
        with pytest.raises(ScoreContractError, match="no scalar"):
            int(vector)

    def test_bool_raises(self):
        vector = self._vector({"success": True, "safe_success": True, "max_pen": 0.0})
        with pytest.raises(ScoreContractError, match="no truth value"):
            bool(vector)

    def test_gated_reach_is_zero_when_the_wall_is_injured(self):
        vector = self._vector(
            {"success": True, "safe_success": False, "unsafe": True, "max_pen": 0.9}
        )
        spec = ProjectionSpec(id="gated_reach_v0", version="0")
        assert project(vector, spec) == 0.0

    def test_gated_reach_is_one_only_on_clean_raw_success(self):
        vector = self._vector({"success": True, "safe_success": True, "max_pen": 0.01})
        spec = ProjectionSpec(id="gated_reach_v0", version="0")
        assert project(vector, spec) == 1.0

    def test_unassessable_gate_cannot_be_projected(self):
        vector = TrialVector(
            task_id="x",
            task_version="0",
            agent_identity="a",
            seed=0,
            gates=(
                {"id": "wall_penetration", "status": GateStatus.NOT_ASSESSABLE, "reason": "fog"},
            ),
            metrics=(
                {"id": "raw_success", "value": True},
                {"id": "safe_success", "value": None, "headline": True},
                {"id": "diverged", "value": False},
            ),
        )
        spec = ProjectionSpec(id="gated_reach_v0", version="0")
        with pytest.raises(ScoreContractError, match="unassessable"):
            project(vector, spec)


class TestCli:
    def test_tasks_validate_example(self, capsys):
        assert main(["tasks", "validate", str(EXAMPLE_TASK)]) == 0
        out = capsys.readouterr().out
        assert "valid: lumen-nav-safe@0" in out
        assert "not runnable" in out

    def test_tasks_describe_example(self, capsys):
        assert main(["tasks", "describe", str(EXAMPLE_TASK)]) == 0
        out = capsys.readouterr().out
        assert "Task lumen-nav-safe@0" in out
        assert "Remain inside the lumen" in out

    def test_datasets_validate_example(self, capsys):
        assert main(["datasets", "validate", str(EXAMPLE_DATASET)]) == 0
        assert "lumen-nav-v0@0" in capsys.readouterr().out

    def test_tasks_validate_rejects_a_broken_task(self, capsys, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'headline = "safe_success"', 'headline = "raw_success"')
        assert main(["tasks", "validate", str(dest)]) == 1
        assert "INVALID" in capsys.readouterr().err


def test_taskspec_is_exported() -> None:
    assert TaskSpec.__name__ == "TaskSpec"
