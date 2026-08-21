"""CLI entry point branding."""

from __future__ import annotations

from pathlib import Path

import pytest

from or_audit.cli import build_parser, main


def test_build_parser_default_prog_is_surgeval() -> None:
    parser = build_parser(prog="surgeval")
    assert parser.prog == "surgeval"


def test_build_parser_vector_alias_prog() -> None:
    parser = build_parser(prog="vector")
    assert parser.prog == "vector"


def test_build_parser_or_audit_alias_prog() -> None:
    parser = build_parser(prog="or-audit")
    assert parser.prog == "or-audit"


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.3.0a0" in capsys.readouterr().out


def test_cloud_worker_command_is_registered() -> None:
    args = build_parser(prog="surgeval").parse_args(["cloud", "worker"])
    assert args.cloud_command == "worker"


def test_cloud_local_execution_refuses_network_bind(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["cloud", "serve", "--enable-local", "--host", "0.0.0.0"]) == 2
    assert "loopback" in capsys.readouterr().err


def test_cloud_serve_builds_local_development_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    seen: dict[str, object] = {}

    def fake_run(app: object, *, host: str, port: int) -> None:
        seen.update(app=app, host=host, port=port)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.delenv("VECTOR_CLOUD_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    root = tmp_path / "vector-cloud"

    assert (
        main(
            [
                "cloud",
                "serve",
                "--enable-local",
                "--allow-anonymous",
                "--port",
                "9876",
                "--db",
                str(root / "jobs.sqlite"),
                "--data",
                str(root / "jobs"),
            ]
        )
        == 0
    )
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 9876
