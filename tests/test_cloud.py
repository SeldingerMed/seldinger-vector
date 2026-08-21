from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from urllib.request import Request

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from or_audit.cloud import worker
from or_audit.cloud.api import create_app
from or_audit.cloud.executors import LocalExecutor, RunPodExecutor
from or_audit.cloud.models import (
    ComputeClass,
    DataClassification,
    ExecutorKind,
    JobRecord,
    JobRequest,
    JobStatus,
)
from or_audit.cloud.store import JobStore
from or_audit.errors import TaskContractError

ROOT = Path(__file__).resolve().parents[1]
VIDEO_TASK = ROOT / "docs" / "examples" / "tasks" / "video-nextstep"
VIDEO_AGENT = ROOT / "docs" / "examples" / "agents" / "example-video-predictor"
PINNED_IMAGE = "registry.example/vector-worker@sha256:" + "a" * 64


class RecordingExecutor:
    def __init__(self) -> None:
        self.submitted: list[str] = []

    def submit(self, job: JobRecord) -> None:
        self.submitted.append(job.id)

    def cancel(self, job: JobRecord) -> None:
        del job

    def reconcile(self, job: JobRecord) -> JobRecord:
        return job


def test_store_persists_job_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite"
    request = JobRequest(task="task", agent="agent")
    created = JobStore(path).create(request)

    loaded = JobStore(path).get(created.id)

    assert loaded is not None
    assert loaded.request == request
    assert loaded.status is JobStatus.QUEUED


def test_terminal_transition_is_compare_and_set(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    record = store.create(JobRequest(task="task", agent="agent"))
    store.transition(
        record.id,
        expected=(JobStatus.QUEUED,),
        status=JobStatus.CANCELLED,
    )

    with pytest.raises(TaskContractError, match="state changed"):
        store.transition(
            record.id,
            expected=(JobStatus.QUEUED, JobStatus.RUNNING),
            status=JobStatus.SUCCEEDED,
        )


def test_hosted_request_refuses_phi_confidential_and_unversioned_packages() -> None:
    with pytest.raises(ValidationError, match="data_classification"):
        JobRequest.model_validate({"task": "t", "agent": "a", "data_classification": "phi"})
    with pytest.raises(ValidationError, match="public or deidentified"):
        JobRequest(
            task="org/task@1",
            agent="org/agent@1",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
            data_classification=DataClassification.CONFIDENTIAL,
        )
    with pytest.raises(ValidationError, match="versioned registry"):
        JobRequest(
            task="local-task",
            agent="local-agent",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )


def test_runpod_executor_refuses_mutable_worker_image(tmp_path: Path) -> None:
    with pytest.raises(TaskContractError, match="sha256"):
        RunPodExecutor(
            JobStore(tmp_path / "jobs.sqlite"),
            api_key="token",
            callback_url="https://vector.example",
            worker_image="registry.example/vector-worker:latest",
        )


def test_api_requires_bearer_token_and_persists_submission(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    executor = RecordingExecutor()
    app = create_app(
        store=store,
        executors={ExecutorKind.LOCAL: executor},
        artifact_root=tmp_path / "data",
        token="secret-token",
    )
    client = TestClient(app)
    payload = {"task": "task", "agent": "agent", "n": 2}

    assert client.post("/v1/jobs", json=payload).status_code == 401
    response = client.post(
        "/v1/jobs",
        json=payload,
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 202
    job = response.json()
    assert executor.submitted == [job["id"]]
    assert (
        client.get(
            f"/v1/jobs/{job['id']}",
            headers={"Authorization": "Bearer secret-token"},
        ).status_code
        == 200
    )
    assert len(JobStore(tmp_path / "jobs.sqlite").list()) == 1


def test_local_executor_runs_real_cli_and_writes_result(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    completed = _run_local_video(store, tmp_path)

    assert completed.status is JobStatus.SUCCEEDED, completed.error
    assert completed.result_head
    assert Path(completed.artifact_path, "result.json").is_file()


def test_local_executor_persists_unexpected_failure(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    executor = LocalExecutor(store, root=tmp_path / "data", package_root=ROOT)
    record = store.create(JobRequest(task="does-not-exist", agent="random"))

    executor.submit(record)
    completed = _wait_for_terminal(store, record.id)

    assert completed.status is JobStatus.FAILED
    assert completed.error


def test_runpod_executor_sends_secure_allowlisted_worker_request(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    seen: list[tuple[str, str, dict[str, object]]] = []

    def transport(request: Request, timeout: float) -> tuple[int, bytes]:
        assert timeout == 30
        assert request.get_header("Authorization") == "Bearer runpod-token"
        assert request.get_header("User-agent") == "VectorCloud/0.1"
        assert isinstance(request.data, bytes)
        body = json.loads(request.data)
        assert isinstance(body, dict)
        seen.append((request.get_method(), request.full_url, body))
        return 201, b'{"id":"pod_123","status":"PROVISIONING"}'

    executor = RunPodExecutor(
        store,
        api_key="runpod-token",
        callback_url="https://vector.example",
        worker_image=PINNED_IMAGE,
        registry_id="reg_private",
        transport=transport,
    )
    record = store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L40S,
            data_classification=DataClassification.DEIDENTIFIED,
        )
    )

    executor.submit(record)
    updated = store.get(record.id)
    body = seen[0][2]
    env = body["env"]

    assert updated is not None
    assert updated.status is JobStatus.PROVISIONING
    assert updated.provider_id == "pod_123"
    assert seen[0][0:2] == ("POST", "https://api.runpod.io/v2/pods")
    assert body["cloud"] == "SECURE"
    assert body["image"] == PINNED_IMAGE
    assert body["args"] == "cloud worker"
    assert body["gpu"] == {"id": "NVIDIA L40S", "count": 1}
    assert body["registry"] == "reg_private"
    assert isinstance(env, dict)
    callback_token = env["VECTOR_CLOUD_CALLBACK_TOKEN"]
    assert isinstance(callback_token, str)
    assert store.verify_callback_token(record.id, callback_token)


def test_remote_callback_can_complete_before_provisioning_and_only_once(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    local = _run_local_video(store, tmp_path)
    remote = store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )
    )
    store.set_callback_token(remote.id, "one-time-token")
    app = create_app(
        store=store,
        executors={},
        artifact_root=tmp_path / "remote-data",
        token="control-token",
    )
    client = TestClient(app)
    archive = _archive(Path(local.artifact_path))
    headers = {
        "Authorization": "Bearer one-time-token",
        "X-Vector-Result-Head": local.result_head,
        "Content-Type": "application/gzip",
    }

    response = client.post(
        f"/v1/internal/jobs/{remote.id}/complete",
        content=archive,
        headers=headers,
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "succeeded"
    assert not store.verify_callback_token(remote.id, "one-time-token")
    assert (
        client.post(
            f"/v1/internal/jobs/{remote.id}/complete",
            content=archive,
            headers=headers,
        ).status_code
        == 401
    )
    result = client.get(
        f"/v1/jobs/{remote.id}/result",
        headers={"Authorization": "Bearer control-token"},
    )
    assert result.status_code == 200


def test_remote_failure_callback_is_job_scoped_and_terminal(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    remote = store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )
    )
    store.set_callback_token(remote.id, "failure-token")
    client = TestClient(
        create_app(
            store=store,
            executors={},
            artifact_root=tmp_path / "data",
            token="control-token",
        )
    )

    response = client.post(
        f"/v1/internal/jobs/{remote.id}/fail",
        json={"error": "model process exited 7"},
        headers={"Authorization": "Bearer failure-token"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error"] == "model process exited 7"
    assert not store.verify_callback_token(remote.id, "failure-token")


def test_remote_callback_rejects_unsafe_archive_without_consuming_token(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    remote = store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )
    )
    store.set_callback_token(remote.id, "archive-token")
    client = TestClient(
        create_app(
            store=store,
            executors={},
            artifact_root=tmp_path / "data",
            token="control-token",
        )
    )
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))

    response = client.post(
        f"/v1/internal/jobs/{remote.id}/complete",
        content=output.getvalue(),
        headers={
            "Authorization": "Bearer archive-token",
            "X-Vector-Result-Head": "a" * 64,
            "Content-Type": "application/gzip",
        },
    )

    assert response.status_code == 422
    assert store.verify_callback_token(remote.id, "archive-token")
    assert not (tmp_path / "escape").exists()


def test_api_configuration_fails_closed_without_token(tmp_path: Path) -> None:
    with pytest.raises(TaskContractError, match="VECTOR_CLOUD_TOKEN"):
        create_app(
            store=JobStore(tmp_path / "jobs.sqlite"),
            executors={},
            artifact_root=tmp_path / "data",
        )


def test_runpod_reconcile_and_cancel_are_terminal_compare_and_set(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    calls: list[str] = []
    provider_status = "RUNNING"

    def transport(request: Request, timeout: float) -> tuple[int, bytes]:
        del timeout
        calls.append(request.get_method())
        if request.get_method() == "POST":
            return 201, b'{"id":"pod_123","status":"PROVISIONING"}'
        if request.get_method() == "GET":
            return 200, json.dumps({"id": "pod_123", "status": provider_status}).encode()
        return 204, b""

    executor = RunPodExecutor(
        store,
        api_key="token",
        callback_url="https://vector.example",
        worker_image=PINNED_IMAGE,
        transport=transport,
    )
    record = store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )
    )
    executor.submit(record)
    running = executor.reconcile(store.get(record.id) or record)
    assert running.status is JobStatus.RUNNING

    executor.cancel(running)
    cancelled = store.get(record.id)
    assert cancelled is not None
    assert cancelled.status is JobStatus.CANCELLED
    assert calls == ["POST", "GET", "DELETE"]


def test_runpod_submit_failure_clears_job_callback_token(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    callback_token = ""

    def transport(request: Request, timeout: float) -> tuple[int, bytes]:
        nonlocal callback_token
        del timeout
        assert isinstance(request.data, bytes)
        body = json.loads(request.data)
        callback_token = body["env"]["VECTOR_CLOUD_CALLBACK_TOKEN"]
        return 500, b"provider unavailable"

    executor = RunPodExecutor(
        store,
        api_key="token",
        callback_url="https://vector.example",
        worker_image=PINNED_IMAGE,
        transport=transport,
    )
    record = store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )
    )

    with pytest.raises(TaskContractError, match="HTTP 500"):
        executor.submit(record)
    assert callback_token
    assert not store.verify_callback_token(record.id, callback_token)


def test_worker_runs_job_and_posts_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    completed = _run_local_video(store, tmp_path)
    source = Path(completed.artifact_path)
    posted: dict[str, object] = {}
    commands: list[list[str]] = []
    _set_worker_env(monkeypatch)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        commands.append(command)
        destination = Path(command[command.index("--out") + 1]) / source.name
        shutil.copytree(source, destination)
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_post(url: str, token: str, payload: bytes, head: str) -> None:
        posted.update(url=url, token=token, payload=payload, head=head)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(worker, "_post_archive", fake_post)

    assert worker.run_from_env() == 0
    assert commands[0][commands[0].index("run") + 1 :][:2] == [
        "-s",
        "seldingermed/video-nextstep@0",
    ]
    assert posted["token"] == "callback-token"
    assert posted["head"] == completed.result_head
    assert str(posted["url"]).endswith("/v1/internal/jobs/job-123/complete")
    assert isinstance(posted["payload"], bytes)


def test_worker_refuses_multi_task_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    posted: dict[str, object] = {}
    _set_worker_env(monkeypatch)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        destination = Path(command[command.index("--out") + 1])
        for task_id in ("first", "second"):
            result = destination / task_id / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_post(url: str, token: str, body: dict[str, str]) -> None:
        posted.update(url=url, token=token, body=body)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(worker, "_post_json", fake_post)

    assert worker.run_from_env() == 1
    assert posted["body"] == {"error": "worker expected one task result, found 2"}


def test_worker_reports_cli_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posted: dict[str, object] = {}
    _set_worker_env(monkeypatch)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(command, 7, "", "model failed")

    def fake_post(url: str, token: str, body: dict[str, str]) -> None:
        posted.update(url=url, token=token, body=body)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(worker, "_post_json", fake_post)

    assert worker.run_from_env() == 7
    assert posted["body"] == {"error": "model failed"}
    assert str(posted["url"]).endswith("/v1/internal/jobs/job-123/fail")


def test_worker_refuses_missing_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VECTOR_CLOUD_CALLBACK_URL",
        "VECTOR_CLOUD_CALLBACK_TOKEN",
        "VECTOR_CLOUD_JOB_ID",
        "VECTOR_CLOUD_TASK",
        "VECTOR_CLOUD_AGENT",
    ):
        monkeypatch.delenv(name, raising=False)
    assert worker.run_from_env() == 2


def _set_worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "VECTOR_CLOUD_CALLBACK_URL": "https://vector.example",
        "VECTOR_CLOUD_CALLBACK_TOKEN": "callback-token",
        "VECTOR_CLOUD_JOB_ID": "job-123",
        "VECTOR_CLOUD_TASK": "seldingermed/video-nextstep@0",
        "VECTOR_CLOUD_AGENT": "example/video-predictor@0",
        "VECTOR_CLOUD_N": "1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _run_local_video(store: JobStore, tmp_path: Path) -> JobRecord:
    executor = LocalExecutor(store, root=tmp_path / "data", package_root=ROOT)
    record = store.create(JobRequest(task=str(VIDEO_TASK), agent=str(VIDEO_AGENT), n=1))
    executor.submit(record)
    return _wait_for_terminal(store, record.id)


def _archive(result_dir: Path) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        archive.add(result_dir, arcname="result")
    return output.getvalue()


def _wait_for_terminal(store: JobStore, job_id: str) -> JobRecord:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        record = store.get(job_id)
        assert record is not None
        if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return record
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not complete")
