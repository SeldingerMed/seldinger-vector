"""Public seldinger-tasks registry contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from or_audit.eval.registry import (
    DEFAULT_REGISTRY,
    _registry_cache_root,
    load_registry,
    resolve_entry,
)


def test_default_registry_index_loads() -> None:
    index = load_registry(DEFAULT_REGISTRY)
    assert index.format_version == "1"
    taskset_refs = {entry.reference for entry in index.tasksets}
    agent_refs = {entry.reference for entry in index.agents}
    assert "seldingermed/video-nextstep@0" in taskset_refs
    assert "example/video-predictor@0" in agent_refs


def test_default_registry_resolves_video_nextstep() -> None:
    index = load_registry(DEFAULT_REGISTRY)
    entry = resolve_entry(index, kind="taskset", ref="seldingermed/video-nextstep@0")
    assert entry.path == "datasets/seldingermed/video-nextstep/0"
    assert entry.repository.endswith("seldinger-tasks.git")


def test_registry_cache_root_prefers_vector_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector_cache = tmp_path / "vector" / "registry"
    legacy_cache = tmp_path / "legacy" / "registry"
    monkeypatch.setattr("or_audit.eval.registry.REGISTRY_CACHE_ROOT", vector_cache)
    monkeypatch.setattr("or_audit.eval.registry.LEGACY_REGISTRY_CACHE_ROOT", legacy_cache)
    assert _registry_cache_root() == vector_cache

    legacy_cache.mkdir(parents=True)
    assert _registry_cache_root() == legacy_cache
