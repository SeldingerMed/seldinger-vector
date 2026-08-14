"""Audit-chain contract from PLAN.md section 7.3.

The chain's whole job is to make retroactive edits detectable, so most of
these tests mutate a valid chain and assert that verification catches it.
"""

from __future__ import annotations

import pytest

from or_audit.audit.trail import (
    GENESIS_HASH,
    Actor,
    ActorKind,
    AuditAction,
    AuditTrail,
    verify_entries,
)
from or_audit.errors import AuditChainError


@pytest.fixture
def trail(frozen_clock, actor) -> AuditTrail:
    built = AuditTrail(clock=frozen_clock)
    built.append(
        actor=actor,
        action=AuditAction.EPISODE_REGISTERED,
        subject_ref="epi_00000000000000000000000000",
        payload={"platform": "hugo"},
    )
    built.append(
        actor=actor,
        action=AuditAction.DEID_ATTESTED,
        subject_ref="epi_00000000000000000000000000",
        payload={"assets": 2},
    )
    return built


class TestChainConstruction:
    def test_empty_trail_head_is_genesis(self, frozen_clock):
        assert AuditTrail(clock=frozen_clock).head_hash == GENESIS_HASH

    def test_first_entry_links_to_genesis(self, trail):
        assert trail.entries[0].prev_hash == GENESIS_HASH

    def test_each_entry_links_to_its_predecessor(self, trail):
        assert trail.entries[1].prev_hash == trail.entries[0].entry_hash

    def test_sequence_is_dense_and_zero_based(self, trail):
        assert [e.seq for e in trail.entries] == [0, 1]

    def test_head_hash_tracks_last_entry(self, trail):
        assert trail.head_hash == trail.entries[-1].entry_hash

    def test_valid_chain_verifies(self, trail):
        trail.verify()

    def test_len_and_iteration(self, trail):
        assert len(trail) == 2
        assert [e.action for e in trail] == [
            AuditAction.EPISODE_REGISTERED,
            AuditAction.DEID_ATTESTED,
        ]

    def test_identical_payloads_produce_distinct_entry_hashes(self, frozen_clock, actor):
        """Sequence and prev_hash must disambiguate otherwise-identical events."""
        built = AuditTrail(clock=frozen_clock)
        for _ in range(2):
            built.append(
                actor=actor,
                action=AuditAction.SCORE_COMPUTED,
                subject_ref="epi_00000000000000000000000000",
                payload={"same": True},
            )
        assert built.entries[0].entry_hash != built.entries[1].entry_hash


class TestTamperDetection:
    def test_payload_edit_is_detected(self, trail):
        tampered = list(trail.entries)
        tampered[0] = tampered[0].model_copy(update={"payload": {"platform": "da_vinci_5"}})
        with pytest.raises(AuditChainError, match="payload does not match"):
            verify_entries(tampered)

    def test_action_edit_is_detected(self, trail):
        tampered = list(trail.entries)
        tampered[1] = tampered[1].model_copy(update={"action": AuditAction.DEID_FAILED})
        with pytest.raises(AuditChainError, match="hash does not match"):
            verify_entries(tampered)

    def test_subject_edit_is_detected(self, trail):
        tampered = list(trail.entries)
        tampered[0] = tampered[0].model_copy(
            update={"subject_ref": "epi_11111111111111111111111111"}
        )
        with pytest.raises(AuditChainError, match="hash does not match"):
            verify_entries(tampered)

    def test_actor_edit_is_detected(self, trail):
        tampered = list(trail.entries)
        tampered[0] = tampered[0].model_copy(
            update={"actor": Actor(kind=ActorKind.OPERATOR, ref="someone-else")}
        )
        with pytest.raises(AuditChainError, match="hash does not match"):
            verify_entries(tampered)

    def test_timestamp_edit_is_detected(self, trail):
        tampered = list(trail.entries)
        moved = tampered[0].recorded_at.replace(year=2030)
        tampered[0] = tampered[0].model_copy(update={"recorded_at": moved})
        with pytest.raises(AuditChainError, match="hash does not match"):
            verify_entries(tampered)

    def test_deleting_an_entry_is_detected(self, trail):
        """Removing the middle of a chain must not verify."""
        with pytest.raises(AuditChainError, match="declares seq"):
            verify_entries(list(trail.entries)[1:])

    def test_reordering_entries_is_detected(self, trail):
        with pytest.raises(AuditChainError, match="declares seq"):
            verify_entries(list(reversed(trail.entries)))

    def test_broken_link_is_detected(self, trail):
        tampered = list(trail.entries)
        forged = tampered[1].model_copy(update={"prev_hash": GENESIS_HASH})
        # Re-derive the entry hash so only the link itself is inconsistent.
        _, entry_hash = forged.recompute_hashes()
        tampered[1] = forged.model_copy(update={"entry_hash": entry_hash})
        with pytest.raises(AuditChainError, match="prev_hash does not match"):
            verify_entries(tampered)

    def test_foreign_chain_version_is_rejected(self, trail):
        tampered = list(trail.entries)
        forged = tampered[0].model_copy(update={"chain_version": "99"})
        _, entry_hash = forged.recompute_hashes()
        tampered[0] = forged.model_copy(update={"entry_hash": entry_hash})
        with pytest.raises(AuditChainError, match="chain version"):
            verify_entries(tampered)


class TestPersistence:
    def test_jsonl_roundtrip_preserves_and_verifies(self, trail, tmp_path):
        path = tmp_path / "audit.jsonl"
        trail.to_jsonl(path)
        loaded = AuditTrail.from_jsonl(path)
        assert [e.entry_hash for e in loaded] == [e.entry_hash for e in trail]
        assert loaded.head_hash == trail.head_hash

    def test_loading_a_tampered_file_raises(self, trail, tmp_path):
        path = tmp_path / "audit.jsonl"
        trail.to_jsonl(path)
        text = path.read_text(encoding="utf-8").replace('"hugo"', '"da_vinci_5"')
        path.write_text(text, encoding="utf-8")
        with pytest.raises(AuditChainError):
            AuditTrail.from_jsonl(path)

    def test_tampered_file_loadable_with_verification_disabled(self, trail, tmp_path):
        """Inspecting a known-broken log must still be possible."""
        path = tmp_path / "audit.jsonl"
        trail.to_jsonl(path)
        text = path.read_text(encoding="utf-8").replace('"hugo"', '"da_vinci_5"')
        path.write_text(text, encoding="utf-8")
        loaded = AuditTrail.from_jsonl(path, verify=False)
        assert len(loaded) == 2
        with pytest.raises(AuditChainError):
            loaded.verify()

    def test_blank_lines_are_tolerated(self, trail, tmp_path):
        path = tmp_path / "audit.jsonl"
        trail.to_jsonl(path)
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        assert len(AuditTrail.from_jsonl(path)) == 2


class TestPayloadHandling:
    def test_absent_payload_becomes_empty_mapping(self, frozen_clock, actor):
        built = AuditTrail(clock=frozen_clock)
        entry = built.append(
            actor=actor,
            action=AuditAction.EXPORT_DENIED,
            subject_ref="epi_00000000000000000000000000",
        )
        assert entry.payload == {}
        built.verify()

    def test_payload_is_snapshotted_not_aliased(self, frozen_clock, actor):
        """A later mutation of the caller's dict must not invalidate the chain."""
        built = AuditTrail(clock=frozen_clock)
        payload = {"n": 1}
        built.append(
            actor=actor,
            action=AuditAction.SCORE_COMPUTED,
            subject_ref="epi_00000000000000000000000000",
            payload=payload,
        )
        payload["n"] = 2
        built.verify()
        assert built.entries[0].payload == {"n": 1}

    def test_uncanonicalizable_payload_is_rejected(self, frozen_clock, actor):
        built = AuditTrail(clock=frozen_clock)
        with pytest.raises(ValueError, match="non-finite"):
            built.append(
                actor=actor,
                action=AuditAction.SCORE_COMPUTED,
                subject_ref="epi_00000000000000000000000000",
                payload={"score": float("nan")},
            )
        assert len(built) == 0, "a rejected append must not leave a partial entry"
